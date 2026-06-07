"""
trajectory_primitives.py
------------------------
Low-level motion primitives for scripted Kinova Gen3 data collection.

Provides:
  ReachPDParams         — configuration dataclass for the PD reach controller
  reach_pose_pd()       — Cartesian PD control loop that drives the arm to a
                          target EE pose, with a virtual-Z transit phase for
                          safe long-distance moves
  send_gripper_binary_state() — open or close the gripper and wait for
                                confirmation
  move_and_grasp()      — reach a pose then close the gripper
  move_and_release()    — reach a pose then open the gripper
  move_and_lift()       — reach a lift pose (gripper unchanged)
  move_and_grasp_then_lift() — grasp at a pose, then move to a lift pose
  move_to_home_position()    — execute the robot's saved "Home" action
  log_feedback_sample() — read BaseCyclic feedback and record one logger step
  warmup_camera()       — wait for the first valid camera frame

Virtual-Z transit
-----------------
For large XY moves, driving straight to the target can result in the arm
colliding with the table or the target object during transit.  reach_pose_pd()
implements a three-phase virtual-Z strategy:

  Phase 1 (ascent)  : rise to virtual_travel_z while keeping XY fixed
  Phase 2 (travel)  : move in XY at virtual_travel_z
  Phase 3 (descent) : descend to the final target Z once XY is close enough

The transition thresholds (xy_far_threshold_m, xy_near_threshold_m) create
a hysteresis band to prevent oscillation between phases.
"""

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2

from episode_logger import EpisodeLogger, RealSenseLatestRGB, normalize_gripper
from utilities import DeviceConnection, parseConnectionArguments


TIMEOUT_DURATION = 30.0
HOME_ACTION_NAME = "Home"  # Name of the pre-saved Home action on the robot


class MotionAbortRequested(Exception):
    """Raised when an external stop event (e.g. ESC key) requests motion abort."""


# ---------------------------------------------------------------------------
# PD controller parameters
# ---------------------------------------------------------------------------

@dataclass
class ReachPDParams:
    """PD gains and convergence criteria for reach_pose_pd().

    Linear gains operate in metres; angular gains operate in degrees.
    Twist commands are clamped to max_twist (m/s) and max_angular_twist (rad/s).
    """
    # Linear PD gains
    kp_x: float = 1.2
    kd_x: float = 0.05
    kp_y: float = 1.2
    kd_y: float = 0.05
    kp_z: float = 1.5
    kd_z: float = 0.08
    # Virtual-Z phase uses higher Z gains to ascend quickly.
    kp_z_virtual: float = 2.5
    kd_z_virtual: float = 0.10
    # Orientation PD gains
    kp_theta_x: float = 0.8
    kd_theta_x: float = 0.03
    kp_theta_y: float = 0.8
    kd_theta_y: float = 0.03
    kp_theta_z: float = 0.8
    kd_theta_z: float = 0.03
    # Twist limits
    max_twist: float = 0.1                  # m/s per axis
    max_angular_twist: float = math.pi / 2  # rad/s per axis (≈ 90°/s)
    # Virtual-Z transit
    virtual_travel_z: float = 0.4           # height (m) used during XY transit
    xy_far_threshold_m: float = 0.10        # XY distance at which virtual-Z activates
    xy_near_threshold_m: float = 0.03       # XY distance at which descent begins
    virtual_z_tolerance_m: float = 0.015    # required closeness to virtual_travel_z
    # Convergence
    pos_tolerance_m: float = 0.010          # position convergence band (m)
    ori_tolerance_deg: float = 7.0          # orientation convergence band (deg)
    timeout_s: float = 25.0
    stable_required_cycles: int = 4         # consecutive in-band cycles to declare convergence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def wrap_deg(d: float) -> float:
    """Wrap an angle difference into [-180, 180)."""
    return (d + 180.0) % 360.0 - 180.0


def clamp(v: float, vmin: float, vmax: float) -> float:
    return max(vmin, min(vmax, v))


def check_for_end_or_abort(e: threading.Event, result: dict):
    """Factory for a Kortex action-notification callback."""
    def check(notification, e=e, result=result):
        print("EVENT : " + Base_pb2.ActionEvent.Name(notification.action_event))
        if notification.action_event in (Base_pb2.ACTION_END, Base_pb2.ACTION_ABORT):
            result["event"] = notification.action_event
            e.set()
    return check


def send_zero_twist(base: BaseClient):
    """Send an all-zeros Cartesian twist command to halt the arm."""
    cmd = Base_pb2.TwistCommand()
    cmd.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE
    if hasattr(cmd, "duration"):
        cmd.duration = 0
    base.SendTwistCommand(cmd)


def read_current_pose(base_cyclic: BaseCyclicClient) -> Dict[str, float]:
    """Return the current measured EE pose as a dict."""
    b = base_cyclic.RefreshFeedback().base
    return {
        "x": float(b.tool_pose_x),
        "y": float(b.tool_pose_y),
        "z": float(b.tool_pose_z),
        "theta_x": float(b.tool_pose_theta_x),
        "theta_y": float(b.tool_pose_theta_y),
        "theta_z": float(b.tool_pose_theta_z),
    }


def pose_error(measured: Dict[str, float], target: Dict[str, float]):
    """Compute position and orientation error between *measured* and *target*.

    Returns:
        (ex, ey, ez, etx, ety, etz, pos_err, ori_err)
        where pos_err is the Euclidean distance and ori_err is the
        maximum absolute angular error across the three axes.
    """
    ex = target["x"] - measured["x"]
    ey = target["y"] - measured["y"]
    ez = target["z"] - measured["z"]
    pos_err = math.sqrt(ex * ex + ey * ey + ez * ez)

    etx = wrap_deg(target["theta_x"] - measured["theta_x"])
    ety = wrap_deg(target["theta_y"] - measured["theta_y"])
    etz = wrap_deg(target["theta_z"] - measured["theta_z"])
    ori_err = max(abs(etx), abs(ety), abs(etz))

    return ex, ey, ez, etx, ety, etz, pos_err, ori_err


def log_feedback_sample(
    base_cyclic: BaseCyclicClient,
    logger: Optional[EpisodeLogger],
    camera: Optional[RealSenseLatestRGB],
    arm_camera=None,
):
    """Read one BaseCyclic feedback sample and optionally record it.

    Returns the raw feedback object so callers can inspect it.
    """
    fb = base_cyclic.RefreshFeedback()
    if logger is not None:
        img, t_img = (camera.get_latest() if camera is not None else (None, None))
        arm_img = None
        if arm_camera is not None:
            try:
                arm_img = arm_camera.get_latest_frame()
            except Exception:
                pass
        gr = normalize_gripper(fb.interconnect.gripper_feedback.motor[0].position)
        joint_angles = [act.position for act in fb.actuators]
        logger.log_step(fb.base, gr, joint_angles=joint_angles, img=img, t_img=t_img, arm_img=arm_img)
    return fb


# ---------------------------------------------------------------------------
# Motion primitives
# ---------------------------------------------------------------------------

def move_to_home_position(base: BaseClient, timeout_s: float = TIMEOUT_DURATION):
    """Execute the robot's saved Home action and block until it completes.

    Args:
        base:      Kortex BaseClient.
        timeout_s: Maximum seconds to wait for the action to finish.

    Raises:
        RuntimeError: If the Home action is not found on the robot.
        TimeoutError: If the action does not complete within *timeout_s*.
    """
    base_servo_mode = Base_pb2.ServoingModeInformation()
    base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
    base.SetServoingMode(base_servo_mode)

    action_type = Base_pb2.RequestedActionType()
    action_type.action_type = Base_pb2.REACH_JOINT_ANGLES
    action_list   = base.ReadAllActions(action_type)
    action_handle = None
    for action in action_list.action_list:
        if action.name == HOME_ACTION_NAME:
            action_handle = action.handle
            break

    if action_handle is None:
        raise RuntimeError(f"Action '{HOME_ACTION_NAME}' not found on the robot.")

    e      = threading.Event()
    result = {"event": None}
    handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e, result), Base_pb2.NotificationOptions()
    )
    base.ExecuteActionFromReference(action_handle)
    finished = e.wait(timeout_s)
    base.Unsubscribe(handle)

    if not finished:
        raise TimeoutError(f"Timed out moving to '{HOME_ACTION_NAME}'.")
    if result["event"] == Base_pb2.ACTION_ABORT:
        raise RuntimeError(f"Move to '{HOME_ACTION_NAME}' was aborted by the robot.")


def _build_pd_twist_command(
    errors: Dict[str, float],
    prev_errors: Dict[str, float],
    dt: float,
    params: ReachPDParams,
    use_virtual_z: bool,
) -> Base_pb2.TwistCommand:
    """Compute a clamped PD Cartesian twist command from current and previous errors."""
    dt_safe = max(1e-4, dt)

    dex  = (errors["ex"]  - prev_errors["ex"])  / dt_safe
    dey  = (errors["ey"]  - prev_errors["ey"])  / dt_safe
    dez  = (errors["ez"]  - prev_errors["ez"])  / dt_safe
    detx = wrap_deg(errors["etx"] - prev_errors["etx"]) / dt_safe
    dety = wrap_deg(errors["ety"] - prev_errors["ety"]) / dt_safe
    detz = wrap_deg(errors["etz"] - prev_errors["etz"]) / dt_safe

    kp_z = params.kp_z_virtual if use_virtual_z else params.kp_z
    kd_z = params.kd_z_virtual if use_virtual_z else params.kd_z

    ux  = params.kp_x       * errors["ex"]  + params.kd_x       * dex
    uy  = params.kp_y       * errors["ey"]  + params.kd_y       * dey
    uz  = kp_z              * errors["ez"]  + kd_z              * dez
    utx = params.kp_theta_x * errors["etx"] + params.kd_theta_x * detx
    uty = params.kp_theta_y * errors["ety"] + params.kd_theta_y * dety
    utz = params.kp_theta_z * errors["etz"] + params.kd_theta_z * detz

    cmd = Base_pb2.TwistCommand()
    cmd.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE
    if hasattr(cmd, "duration"):
        cmd.duration = 0

    vl = abs(params.max_twist)
    va = abs(params.max_angular_twist)
    cmd.twist.linear_x  = clamp(ux,  -vl, vl)
    cmd.twist.linear_y  = clamp(uy,  -vl, vl)
    cmd.twist.linear_z  = clamp(uz,  -vl, vl)
    cmd.twist.angular_x = clamp(utx, -va, va)
    cmd.twist.angular_y = clamp(uty, -va, va)
    cmd.twist.angular_z = clamp(utz, -va, va)
    return cmd


def reach_pose_pd(
    base: BaseClient,
    base_cyclic: BaseCyclicClient,
    target_pose: Dict[str, float],
    params: ReachPDParams,
    hz: float = 20.0,
    logger: Optional[EpisodeLogger] = None,
    camera: Optional[RealSenseLatestRGB] = None,
    arm_camera=None,
    stop_event: Optional[threading.Event] = None,
) -> bool:
    """Drive the arm to *target_pose* using a Cartesian PD controller.

    Incorporates a three-phase virtual-Z transit for safe long-range moves
    (see module docstring).  Logs feedback at every control tick if *logger*
    and *camera* are provided.

    Args:
        base, base_cyclic: Kortex client handles.
        target_pose:  Dict with keys x, y, z, theta_x, theta_y, theta_z.
        params:       PD gains and convergence criteria.
        hz:           Control loop frequency.
        logger:       EpisodeLogger instance for recording, or None.
        camera:       Scene camera for image logging, or None.
        arm_camera:   Wrist camera FrameGrabber, or None.
        stop_event:   If set, the loop aborts and raises MotionAbortRequested.

    Returns:
        True if the pose was reached within timeout, False otherwise.
    """
    if params.xy_near_threshold_m > params.xy_far_threshold_m:
        raise ValueError("xy_near_threshold_m must be <= xy_far_threshold_m")
    if params.virtual_z_tolerance_m < 0.0:
        raise ValueError("virtual_z_tolerance_m must be >= 0")

    dt = 1.0 / max(1e-3, hz)
    t0 = time.time()
    stable_count = 0

    # Virtual-Z state machine:
    #   0 = direct-to-target (no virtual phase)
    #   1 = ascending to virtual_travel_z (XY locked)
    #   2 = XY travel at virtual_travel_z
    virtual_mode      = 0
    virtual_phase_done = False
    ascent_anchor_x   = None
    ascent_anchor_y   = None
    prev_virtual_mode = 0

    prev_errors = {k: 0.0 for k in ("ex", "ey", "ez", "etx", "ety", "etz")}
    have_prev   = False

    while time.time() - t0 < params.timeout_s:
        if stop_event is not None and stop_event.is_set():
            send_zero_twist(base)
            raise MotionAbortRequested("Motion aborted by stop event.")

        loop_t  = time.time()
        fb      = log_feedback_sample(base_cyclic, logger, camera, arm_camera=arm_camera)
        measured = {
            "x":       float(fb.base.tool_pose_x),
            "y":       float(fb.base.tool_pose_y),
            "z":       float(fb.base.tool_pose_z),
            "theta_x": float(fb.base.tool_pose_theta_x),
            "theta_y": float(fb.base.tool_pose_theta_y),
            "theta_z": float(fb.base.tool_pose_theta_z),
        }

        ex_f, ey_f, _, etx_f, ety_f, etz_f, pos_err, ori_err = pose_error(measured, target_pose)
        xy_dist       = math.sqrt(ex_f ** 2 + ey_f ** 2)
        virtual_z_err = params.virtual_travel_z - measured["z"]
        virtual_z_rdy = abs(virtual_z_err) <= params.virtual_z_tolerance_m

        # Update virtual-Z phase transitions.
        if not virtual_phase_done:
            if virtual_mode == 0 and xy_dist > params.xy_far_threshold_m:
                if virtual_z_rdy:
                    virtual_mode = 2
                else:
                    virtual_mode = 1
                    ascent_anchor_x = measured["x"]
                    ascent_anchor_y = measured["y"]
            if virtual_mode == 1 and virtual_z_rdy:
                virtual_mode = 2
            if virtual_mode == 2 and xy_dist <= params.xy_near_threshold_m:
                virtual_mode = 0
                virtual_phase_done = True

        # Build the effective control target for the current phase.
        if virtual_mode == 1:
            control_target = {
                "x": float(ascent_anchor_x),
                "y": float(ascent_anchor_y),
                "z": params.virtual_travel_z,
                "theta_x": target_pose["theta_x"],
                "theta_y": target_pose["theta_y"],
                "theta_z": target_pose["theta_z"],
            }
            use_virtual_z = True
        elif virtual_mode == 2:
            control_target = {
                "x": target_pose["x"],
                "y": target_pose["y"],
                "z": params.virtual_travel_z,
                "theta_x": target_pose["theta_x"],
                "theta_y": target_pose["theta_y"],
                "theta_z": target_pose["theta_z"],
            }
            use_virtual_z = True
        else:
            control_target = target_pose
            use_virtual_z  = False

        ex, ey, ez, etx, ety, etz, _, _ = pose_error(measured, control_target)
        errors = {"ex": ex, "ey": ey, "ez": ez, "etx": etx, "ety": ety, "etz": etz}

        if not have_prev:
            prev_errors = dict(errors)
            have_prev   = True

        # Reset derivative term on phase transition to avoid derivative kick.
        if virtual_mode != prev_virtual_mode:
            prev_errors = dict(errors)
        prev_virtual_mode = virtual_mode

        if pos_err <= params.pos_tolerance_m and ori_err <= params.ori_tolerance_deg:
            stable_count += 1
            send_zero_twist(base)
            if stable_count >= params.stable_required_cycles:
                return True
        else:
            stable_count = 0
            cmd = _build_pd_twist_command(errors, prev_errors, dt, params, use_virtual_z)
            base.SendTwistCommand(cmd)

        prev_errors = dict(errors)

        elapsed = time.time() - loop_t
        if dt > elapsed:
            time.sleep(dt - elapsed)

    send_zero_twist(base)
    return False


def send_gripper_binary_state(
    base: BaseClient,
    gripper_pos: int,
    timeout_s: float = 2.5,
    tol: float = 0.1,
    base_cyclic: Optional[BaseCyclicClient] = None,
    logger: Optional[EpisodeLogger] = None,
    camera: Optional[RealSenseLatestRGB] = None,
    arm_camera=None,
    hz: float = 20.0,
    stop_event: Optional[threading.Event] = None,
) -> bool:
    """Command the gripper to a binary open/close state and wait for it to arrive.

    Args:
        gripper_pos: 1 = open, 0 = closed  (dataset convention).
        timeout_s:   Maximum wait time.
        tol:         Acceptable position error for the gripper motor.

    Returns:
        True once the gripper reaches the target (or on timeout).
    """
    if gripper_pos not in (0, 1):
        raise ValueError("gripper_pos must be 0 (closed) or 1 (open)")

    # Robot motor convention is opposite to the dataset convention:
    # motor 0.0 = open, motor 0.8 = closed.
    target_motor = 0.0 if gripper_pos == 1 else 0.8

    cmd = Base_pb2.GripperCommand()
    cmd.mode = Base_pb2.GRIPPER_POSITION
    finger = cmd.gripper.finger.add()
    finger.value = target_motor

    request = Base_pb2.GripperRequest()
    request.mode = Base_pb2.GRIPPER_POSITION

    dt = 1.0 / hz
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if stop_event is not None and stop_event.is_set():
            send_zero_twist(base)
            raise MotionAbortRequested("Gripper action aborted by stop event.")

        loop_t = time.time()
        base.SendGripperCommand(cmd)
        measure = base.GetMeasuredGripperMovement(request)

        if base_cyclic is not None:
            log_feedback_sample(base_cyclic, logger, camera, arm_camera=arm_camera)

        if measure.finger and abs(float(measure.finger[0].value) - target_motor) <= tol:
            return True

        elapsed = time.time() - loop_t
        if dt > elapsed:
            time.sleep(dt - elapsed)

    return True  # Return True on timeout — gripper command was sent


def move_and_grasp(
    base: BaseClient,
    base_cyclic: BaseCyclicClient,
    ee_pose: Dict[str, float],
    reach_params: ReachPDParams,
    hz: float = 20.0,
    logger: Optional[EpisodeLogger] = None,
    camera: Optional[RealSenseLatestRGB] = None,
    arm_camera=None,
    stop_event: Optional[threading.Event] = None,
):
    """Reach *ee_pose* with the arm, then close the gripper."""
    reached = reach_pose_pd(
        base, base_cyclic, ee_pose, params=reach_params,
        hz=hz, logger=logger, camera=camera, arm_camera=arm_camera, stop_event=stop_event,
    )
    if not reached:
        raise RuntimeError("Target pose was not reached before grasp.")
    ok = send_gripper_binary_state(
        base, gripper_pos=0, base_cyclic=base_cyclic,
        logger=logger, camera=camera, arm_camera=arm_camera, hz=hz, stop_event=stop_event,
    )
    if not ok:
        raise RuntimeError("Gripper close command timed out.")


def move_and_release(
    base: BaseClient,
    base_cyclic: BaseCyclicClient,
    ee_pose: Dict[str, float],
    reach_params: ReachPDParams,
    hz: float = 20.0,
    logger: Optional[EpisodeLogger] = None,
    camera: Optional[RealSenseLatestRGB] = None,
    arm_camera=None,
    stop_event: Optional[threading.Event] = None,
):
    """Reach *ee_pose* with the arm, then open the gripper."""
    reached = reach_pose_pd(
        base, base_cyclic, ee_pose, params=reach_params,
        hz=hz, logger=logger, camera=camera, arm_camera=arm_camera, stop_event=stop_event,
    )
    if not reached:
        raise RuntimeError("Target pose was not reached before release.")
    ok = send_gripper_binary_state(
        base, gripper_pos=1, base_cyclic=base_cyclic,
        logger=logger, camera=camera, arm_camera=arm_camera, hz=hz, stop_event=stop_event,
    )
    if not ok:
        raise RuntimeError("Gripper open command timed out.")


def move_and_lift(
    base: BaseClient,
    base_cyclic: BaseCyclicClient,
    lift_pose: Dict[str, float],
    reach_params: ReachPDParams,
    hz: float = 20.0,
    logger: Optional[EpisodeLogger] = None,
    camera: Optional[RealSenseLatestRGB] = None,
    arm_camera=None,
    stop_event: Optional[threading.Event] = None,
):
    """Reach *lift_pose*, leaving the gripper state unchanged."""
    reached = reach_pose_pd(
        base, base_cyclic, lift_pose, params=reach_params,
        hz=hz, logger=logger, camera=camera, arm_camera=arm_camera, stop_event=stop_event,
    )
    if not reached:
        raise RuntimeError("Lift pose was not reached.")


def move_and_grasp_then_lift(
    base: BaseClient,
    base_cyclic: BaseCyclicClient,
    grasp_pose: Dict[str, float],
    lift_pose: Dict[str, float],
    reach_params: ReachPDParams,
    hz: float = 20.0,
    logger: Optional[EpisodeLogger] = None,
    camera: Optional[RealSenseLatestRGB] = None,
    arm_camera=None,
    stop_event: Optional[threading.Event] = None,
):
    """Grasp at *grasp_pose*, then move to *lift_pose*."""
    move_and_grasp(
        base, base_cyclic, grasp_pose, reach_params=reach_params,
        hz=hz, logger=logger, camera=camera, arm_camera=arm_camera, stop_event=stop_event,
    )
    move_and_lift(
        base, base_cyclic, lift_pose, reach_params=reach_params,
        hz=hz, logger=logger, camera=camera, arm_camera=arm_camera, stop_event=stop_event,
    )


def warmup_camera(camera: Optional[RealSenseLatestRGB], timeout_s: float = 1.0):
    """Block until the camera delivers at least one valid frame."""
    if camera is None:
        return
    print("Warming up camera...")
    time.sleep(0.5)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        img, _ = camera.get_latest()
        if img is not None:
            break
        time.sleep(0.03)
    print("Camera ready.")
