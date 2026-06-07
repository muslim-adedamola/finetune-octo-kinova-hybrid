"""
collect_episode.py
------------------
Main data-collection script for the Kinova Gen3 bottle pick-and-lift task.

Pipeline for one episode
------------------------
1. Warm up the RealSense camera.
2. Detect the bottle with YOLOE and project its centre to world coordinates.
3. Plan approach, grasp, and lift poses from the detected bottle position.
4. Connect to the robot, move to Home, then execute:
     Home → approach pose → grasp pose (close gripper) → lift pose
5. Hold the lift pose briefly, then mark the episode as successful.

ESC at any point aborts the motion, returns the arm to Home, and discards
the episode directory.  Only successfully completed episodes are kept.

Usage
-----
    python collect_episode.py [--out_root DIR] [--hz HZ] [OPTIONS]

See --help for the full list of options including PD gains, detection
timeouts, and motion parameters.

Notes for adapting to a new setup
----------------------------------
- Update camera_config.py (intrinsics, extrinsics, BOTTLE_Z_WORLD, shift vector).
- Replace sample_image.png with a reference image of your target object.
- Update VISUAL_PROMPTS with a bounding box drawn around your object in
  the reference image.
- Retune the PD gains if your robot or control frequency differs.
"""

import argparse
import shutil
import threading
import time

import cv2
import numpy as np
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

from camera_config import (
    BOTTLE_Z_WORLD,
    CAMERA_MATRIX,
    DIST_COEFFS,
    SHIFT_VEC_BOTTLES,
    T_BASE_CAMERA,
)
from episode_logger import EpisodeLogger, RealSenseLatestRGB, create_arm_camera_grabber
from trajectory_primitives import (
    MotionAbortRequested,
    ReachPDParams,
    log_feedback_sample,
    move_to_home_position,
    move_and_grasp_then_lift,
    reach_pose_pd,
    warmup_camera,
    HOME_ACTION_NAME,
)
from utilities import DeviceConnection, parseConnectionArguments


# ---------------------------------------------------------------------------
# YOLOE visual prompt
# NOTE: Replace REFERENCE_IMAGE_PATH with your own reference image and
# update VISUAL_PROMPTS with a bounding box drawn around the target object
# in that image (format: [x1, y1, x2, y2] in pixel coordinates).
# ---------------------------------------------------------------------------
REFERENCE_IMAGE_PATH = "sample_image.png" #example reference image used is shown in 

VISUAL_PROMPTS = dict(
    bboxes=np.array([[563, 431, 623, 582]]),
    cls=np.array([0]),
)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def pixel_to_world_on_plane(u: float, v: float, z_world: float):
    """Back-project a pixel (u, v) onto the horizontal plane z = z_world.

    Uses CAMERA_MATRIX, DIST_COEFFS, and T_BASE_CAMERA from camera_config.

    Returns:
        3-element numpy array [x, y, z] in robot base frame, or None if the
        ray does not intersect the plane (e.g., ray points upward).
    """
    uv     = np.array([[[u, v]]], dtype=np.float32)
    undist = cv2.undistortPoints(uv, CAMERA_MATRIX, DIST_COEFFS)
    x_n, y_n = undist[0, 0]

    ray_cam  = np.array([x_n, y_n, 1.0], dtype=np.float64)
    ray_cam /= np.linalg.norm(ray_cam)

    R = T_BASE_CAMERA[:3, :3]
    t = T_BASE_CAMERA[:3,  3]
    ray_base = R @ ray_cam

    if abs(ray_base[2]) < 1e-9:
        return None
    s = (z_world - t[2]) / ray_base[2]
    if s <= 0:
        return None
    return t + s * ray_base


def detect_bottle(camera: RealSenseLatestRGB, model: YOLOE):
    """Run YOLOE on the latest camera frame and return the best detection.

    Returns a dict with keys:
        detected_cls, conf, world (np.array), bbox_xyxy, uv
    or None if no bottle is detected.
    """
    img, _ = camera.get_latest()
    if img is None:
        return None

    result = model.predict(
        source=img,
        refer_image=REFERENCE_IMAGE_PATH,
        visual_prompts=VISUAL_PROMPTS,
        predictor=YOLOEVPSegPredictor,
        conf=0.25,
        verbose=False,
    )[0]

    if result is None:
        return None

    best = None
    for box in (result.boxes or []):
        cls_id = int(box.cls.item())
        conf   = float(box.conf.item())
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
        u = 0.5 * (x1 + x2)
        v = 0.5 * (y1 + y2)

        p_world = pixel_to_world_on_plane(u, v, BOTTLE_Z_WORLD)
        if p_world is None:
            continue
        p_world = p_world + SHIFT_VEC_BOTTLES

        cand = {
            "detected_cls": cls_id,
            "conf":         conf,
            "world":        p_world,
            "bbox_xyxy":    [x1, y1, x2, y2],
            "uv":           [u, v],
        }
        if best is None or cand["conf"] > best["conf"]:
            best = cand

    return best


# ---------------------------------------------------------------------------
# Pose planning
# ---------------------------------------------------------------------------

def make_approach_pose(world_xyz, approach_height: float) -> dict:
    return {
        "x": float(world_xyz[0]),
        "y": float(world_xyz[1]),
        "z": float(world_xyz[2] + approach_height),
        "theta_x":  90.93,
        "theta_y":  -0.53,
        "theta_z":  91.17,
    }


def make_grasp_pose(world_xyz, grasp_z_offset: float) -> dict:
    return {
        "x": float(world_xyz[0]),
        "y": float(world_xyz[1]),
        "z": float(world_xyz[2] + grasp_z_offset),
        "theta_x":  90.93,
        "theta_y":  -0.53,
        "theta_z":  91.17,
    }


def make_lift_pose(world_xyz, lift_height: float) -> dict:
    return {
        "x": float(world_xyz[0]),
        "y": float(world_xyz[1]),
        "z": float(world_xyz[2] + lift_height),
        "theta_x":  90.93,
        "theta_y":  -0.53,
        "theta_z":  91.17,
    }


# ---------------------------------------------------------------------------
# ESC abort helpers
# ---------------------------------------------------------------------------

def _getch():
    try:
        import sys, tty, termios
        fd  = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        return ch
    except Exception:
        try:
            import msvcrt
            return msvcrt.getch().decode("utf-8", errors="ignore")
        except Exception:
            return None


def esc_listener(stop_event: threading.Event):
    """Background thread: set *stop_event* when ESC is pressed."""
    while not stop_event.is_set():
        ch = _getch()
        if ch is None:
            time.sleep(0.05)
            continue
        if ch == "\x1b":
            stop_event.set()
            print("\nESC pressed — aborting episode.")
            break


def discard_episode(logger: EpisodeLogger):
    """Close the logger and delete the episode directory."""
    try:
        if hasattr(logger, "_f") and logger._f is not None and not logger._f.closed:
            logger._f.flush()
            logger._f.close()
    except Exception:
        pass
    shutil.rmtree(logger.ep_dir, ignore_errors=True)


def stop_and_home_via_new_connection(args):
    """Open a fresh TCP connection, stop the arm, and move it to Home.

    Used in the ESC-abort path where the original connection may already
    be in an error state.
    """
    try:
        with DeviceConnection.createTcpConnection(args) as router_tcp:
            from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
            base = BaseClient(router_tcp)
            try:
                base.Stop()
            except Exception:
                pass
            move_to_home_position(base)
    except Exception as exc:
        print(f"Warning: stop/home on ESC fallback failed: {exc}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Vision-guided bottle pick-and-lift data collection for Kinova Gen3."
    )
    # Episode settings
    parser.add_argument("--out_root",    type=str,   default="kinova_bottle_lift_raw",
                        help="Root directory for episode output (default: kinova_bottle_lift_raw)")
    parser.add_argument("--hz",          type=float, default=10.0,
                        help="Control and logging frequency in Hz (default: 10)")
    parser.add_argument("--no_log_images", action="store_true",
                        help="Disable image saving to reduce disk usage")
    parser.add_argument("--no_arm_camera", action="store_true",
                        help="Disable wrist camera capture")

    # Motion parameters
    parser.add_argument("--approach_height", type=float, default=0.0,
                        help="Z offset above bottle world Z for the approach pose (m)")
    parser.add_argument("--grasp_z_offset",  type=float, default=0.00,
                        help="Z offset added to bottle world Z for the grasp pose (m)")
    parser.add_argument("--lift_height",     type=float, default=0.20,
                        help="Z offset above bottle world Z for the lift pose (m)")
    parser.add_argument("--hold_time",       type=float, default=2.0,
                        help="Seconds to hold the lift pose after grasping")
    parser.add_argument("--detect_timeout",  type=float, default=15.0,
                        help="Seconds to wait for a bottle detection before aborting")

    # PD gains
    parser.add_argument("--kp_x",           type=float, default=1.2)
    parser.add_argument("--kd_x",           type=float, default=0.05)
    parser.add_argument("--kp_y",           type=float, default=1.2)
    parser.add_argument("--kd_y",           type=float, default=0.05)
    parser.add_argument("--kp_z",           type=float, default=1.5)
    parser.add_argument("--kd_z",           type=float, default=0.08)
    parser.add_argument("--kp_z_virtual",   type=float, default=5.0)
    parser.add_argument("--kd_z_virtual",   type=float, default=0.10)
    parser.add_argument("--kp_theta_x",     type=float, default=0.8)
    parser.add_argument("--kd_theta_x",     type=float, default=0.03)
    parser.add_argument("--kp_theta_y",     type=float, default=0.8)
    parser.add_argument("--kd_theta_y",     type=float, default=0.03)
    parser.add_argument("--kp_theta_z",     type=float, default=0.8)
    parser.add_argument("--kd_theta_z",     type=float, default=0.03)
    parser.add_argument("--max_twist",      type=float, default=0.1,
                        help="Linear twist clamp per axis (m/s)")
    parser.add_argument("--max_angular_twist", type=float, default=np.pi / 2,
                        help="Angular twist clamp per axis (rad/s)")
    parser.add_argument("--virtual_travel_z",  type=float, default=0.4,
                        help="Virtual Z height used during long XY moves (m)")
    parser.add_argument("--xy_far_threshold",  type=float, default=0.15,
                        help="XY distance at which virtual-Z transit activates (m)")
    parser.add_argument("--xy_near_threshold", type=float, default=0.15,
                        help="XY distance at which descent to final Z begins (m)")
    parser.add_argument("--virtual_z_tol",     type=float, default=0.1,
                        help="Required closeness to virtual_travel_z before XY travel (m)")
    parser.add_argument("--reach_timeout",     type=float, default=25.0)
    parser.add_argument("--pos_tol",           type=float, default=0.015,
                        help="Position convergence tolerance (m)")
    parser.add_argument("--ori_tol",           type=float, default=7.0,
                        help="Orientation convergence tolerance (deg)")
    parser.add_argument("--stable_cycles",     type=int,   default=4,
                        help="Consecutive in-tolerance cycles to declare convergence")
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = build_arg_parser()
    args   = parseConnectionArguments(parser=parser)

    if args.xy_near_threshold > args.xy_far_threshold:
        raise ValueError("--xy_near_threshold must be <= --xy_far_threshold")
    if args.virtual_z_tol < 0.0:
        raise ValueError("--virtual_z_tol must be >= 0")

    reach_params = ReachPDParams(
        kp_x=args.kp_x, kd_x=args.kd_x,
        kp_y=args.kp_y, kd_y=args.kd_y,
        kp_z=args.kp_z, kd_z=args.kd_z,
        kp_z_virtual=args.kp_z_virtual, kd_z_virtual=args.kd_z_virtual,
        kp_theta_x=args.kp_theta_x, kd_theta_x=args.kd_theta_x,
        kp_theta_y=args.kp_theta_y, kd_theta_y=args.kd_theta_y,
        kp_theta_z=args.kp_theta_z, kd_theta_z=args.kd_theta_z,
        max_twist=args.max_twist,
        max_angular_twist=args.max_angular_twist,
        virtual_travel_z=args.virtual_travel_z,
        xy_far_threshold_m=args.xy_far_threshold,
        xy_near_threshold_m=args.xy_near_threshold,
        virtual_z_tolerance_m=args.virtual_z_tol,
        pos_tolerance_m=args.pos_tol,
        ori_tolerance_deg=args.ori_tol,
        timeout_s=args.reach_timeout,
        stable_required_cycles=args.stable_cycles,
    )

    logger      = EpisodeLogger(out_root=args.out_root, hz=args.hz,
                                save_arm_camera=not args.no_arm_camera)
    camera      = RealSenseLatestRGB(width=1280, height=720, fps=30)
    model       = YOLOE("yoloe-26x-seg.pt")
    log_camera  = None if args.no_log_images else camera

    arm_camera  = None
    if (not args.no_log_images) and (not args.no_arm_camera):
        try:
            arm_camera = create_arm_camera_grabber()
        except Exception as exc:
            print(f"Warning: could not start arm camera: {exc}")
    else:
        print("Arm camera disabled.")

    esc_stop   = threading.Event()
    threading.Thread(target=esc_listener, args=(esc_stop,), daemon=True).start()
    print("Press ESC at any time to abort, return to Home, and discard this episode.")

    aborted_by_esc = False

    try:
        warmup_camera(camera)

        # --- Bottle detection ---
        print("Detecting bottle world coordinate...")
        bottle_det = None
        t_detect   = time.time()
        while True:
            if esc_stop.is_set():
                raise MotionAbortRequested("ESC pressed during detection.")
            bottle_det = detect_bottle(camera, model)
            if bottle_det is not None:
                break
            if time.time() - t_detect > args.detect_timeout:
                break
            time.sleep(0.1)

        if bottle_det is None:
            raise RuntimeError(
                f"No bottle detected within {args.detect_timeout:.0f}s. "
                "Check the camera view and YOLOE reference image."
            )

        bottle_xyz   = bottle_det["world"]
        approach_pose = make_approach_pose(bottle_xyz, args.approach_height)
        grasp_pose    = make_grasp_pose(bottle_xyz,    args.grasp_z_offset)
        lift_pose     = make_lift_pose(bottle_xyz,     args.lift_height)

        print(f"Bottle detected at: {bottle_xyz}")
        print(f"Approach pose : {approach_pose}")
        print(f"Grasp pose    : {grasp_pose}")
        print(f"Lift pose     : {lift_pose}")

        # --- Robot motion ---
        with DeviceConnection.createTcpConnection(args) as router_tcp, \
             DeviceConnection.createUdpConnection(args) as router_udp:

            from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
            from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
            from kortex_api.autogen.messages import Base_pb2

            base        = BaseClient(router_tcp)
            base_cyclic = BaseCyclicClient(router_udp)

            servo_mode = Base_pb2.ServoingModeInformation()
            servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
            base.SetServoingMode(servo_mode)

            try:
                if esc_stop.is_set():
                    raise MotionAbortRequested("ESC pressed before motion.")

                log_feedback_sample(base_cyclic, logger, log_camera, arm_camera=arm_camera)
                print(f"Moving to '{HOME_ACTION_NAME}'...")
                move_to_home_position(base)
                time.sleep(0.5)
                log_feedback_sample(base_cyclic, logger, log_camera, arm_camera=arm_camera)

                print("Moving to approach pose...")
                reached = reach_pose_pd(
                    base, base_cyclic, approach_pose, params=reach_params,
                    hz=args.hz, logger=logger, camera=log_camera,
                    arm_camera=arm_camera, stop_event=esc_stop,
                )
                if not reached:
                    raise MotionAbortRequested("Approach pose was not reached.")

                print("Grasping and lifting...")
                move_and_grasp_then_lift(
                    base, base_cyclic, grasp_pose, lift_pose,
                    reach_params=reach_params, hz=args.hz,
                    logger=logger, camera=log_camera,
                    arm_camera=arm_camera, stop_event=esc_stop,
                )

                print(f"Holding lift pose for {args.hold_time:.1f}s...")
                t_hold = time.time()
                while time.time() - t_hold < args.hold_time:
                    if esc_stop.is_set():
                        raise MotionAbortRequested("ESC pressed during hold.")
                    log_feedback_sample(base_cyclic, logger, log_camera, arm_camera=arm_camera)
                    time.sleep(max(0.01, 1.0 / args.hz))

                logger.success = True
                print("Episode completed successfully.")

            except MotionAbortRequested:
                aborted_by_esc = esc_stop.is_set()
                logger.success = False
                raise

            except Exception:
                logger.success = False
                raise

            finally:
                try:
                    base.Stop()
                except Exception:
                    pass

    except MotionAbortRequested as exc:
        print(f"Aborted: {exc}")
        aborted_by_esc = aborted_by_esc or esc_stop.is_set()

    except Exception as exc:
        print(f"Error: {exc}")
        raise

    finally:
        logger.close()

        if arm_camera is not None:
            try:
                arm_camera.proc.terminate()
                arm_camera.proc.wait(timeout=1.0)
            except Exception:
                pass
            try:
                arm_camera.stop()
            except Exception:
                pass

        if camera is not None:
            camera.close()

        if aborted_by_esc:
            print("ESC abort: returning to Home and discarding episode...")
            stop_and_home_via_new_connection(args)
            discard_episode(logger)
            print("Episode discarded.")
        else:
            print(f"Episode saved to: {logger.ep_dir}")


if __name__ == "__main__":
    main()
