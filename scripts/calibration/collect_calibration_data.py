"""
collect_calibration_data.py
---------------------------
Stage 1 of hand-eye (extrinsic) calibration for the Kinova Gen3 +
Intel RealSense setup.

The robot autonomously moves through randomised Cartesian waypoints while
holding a ChArUco board at its end effector.  At each waypoint the script
checks whether the board is clearly visible in the scene camera, and if so
saves the image and the robot EE pose to disk.  The resulting paired
(image, pose) dataset is consumed by run_calibration.py to solve for
T_BASE_CAMERA.

Physical setup
--------------
1. Mount a printed ChArUco board (DICT_6X6_250, 7×5 squares) rigidly to
   the Kinova end effector so that it faces the external RealSense camera.
2. Position the robot so that the board is visible from the camera in its
   initial (reference) pose.
3. Run this script.  The arm will explore random offsets from the initial
   pose, saving frames where the board has at least MIN_CHARUCO_CORNERS
   detected corners.

Output
------
<save_dir>/
  extrinsic_0000.png
  extrinsic_0001.png
  ...
  robot_poses.json      # {filename: {x, y, z, theta_x, theta_y, theta_z}}

Run
---
    python collect_calibration_data.py [--save_dir DIR] [--n_poses N]

See --help for all options.

Notes
-----
- utilities.py (Kortex connection helpers) is expected to be on the Python
  path.  If running from scripts/calibration/, add scripts/deployment/ to
  PYTHONPATH, or copy utilities.py to this directory.
- RealSenseLatestRGB is reproduced here so this script is self-contained.
  It mirrors the version in scripts/collection/episode_logger.py.
"""

import argparse
import json
import math
import os
import random
import threading
import time

import cv2
import numpy as np
import pyrealsense2 as rs

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2

# Import Kortex connection helpers.
# Ensure scripts/deployment/ is on your PYTHONPATH, or copy utilities.py
# to scripts/calibration/.
from utilities import DeviceConnection, parseConnectionArguments


# ---------------------------------------------------------------------------
# ChArUco board parameters
# NOTE: these must match the physical board you printed.
# Measure SQUARE_LENGTH and MARKER_LENGTH on the printed board with calipers.
# ---------------------------------------------------------------------------
SQUARES_X     = 7
SQUARES_Y     = 5
SQUARE_LENGTH = 0.0358   # metres
MARKER_LENGTH = 0.0258   # metres

# Minimum number of ChArUco corners that must be detected for a frame to
# be accepted.  Higher values give better PnP conditioning.
MIN_CHARUCO_CORNERS = 15

TIMEOUT_DURATION = 20   # seconds per Kortex action


# ---------------------------------------------------------------------------
# Threaded RealSense camera
# (mirrors RealSenseLatestRGB in scripts/collection/episode_logger.py)
# ---------------------------------------------------------------------------

class RealSenseLatestRGB:
    """Captures BGR frames from a RealSense camera in a background thread."""

    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.pipeline.start(config)

        self._lock       = threading.Lock()
        self._latest_img = None
        self._latest_t   = None
        self._stop       = False
        self._thread     = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop:
            try:
                frames = self.pipeline.wait_for_frames()
                color  = frames.get_color_frame()
                if not color:
                    continue
                img = np.asanyarray(color.get_data())
                with self._lock:
                    self._latest_img = img
                    self._latest_t   = time.time()
            except Exception:
                continue

    def get_latest(self):
        with self._lock:
            if self._latest_img is None:
                return None, None
            return self._latest_img.copy(), self._latest_t

    def close(self):
        self._stop = True
        try:
            self._thread.join(timeout=1.0)
            self.pipeline.stop()
        except Exception:
            pass


def warmup_camera(camera: RealSenseLatestRGB, timeout: float = 2.5):
    print("Warming up camera...")
    time.sleep(timeout)
    for _ in range(30):
        img, _ = camera.get_latest()
        if img is not None:
            break
        time.sleep(0.03)
    print("Camera ready.")


# ---------------------------------------------------------------------------
# Robot motion
# ---------------------------------------------------------------------------

def _check_for_end_or_abort(event: threading.Event):
    """Kortex notification callback factory."""
    def check(notification, e=event):
        if notification.action_event in (Base_pb2.ACTION_END, Base_pb2.ACTION_ABORT):
            e.set()
    return check


def move_to_cartesian_pose(
    base: BaseClient,
    x: float, y: float, z: float,
    theta_x: float, theta_y: float, theta_z: float,
    timeout_s: float = TIMEOUT_DURATION,
) -> bool:
    """Command the arm to a Cartesian EE pose and block until it arrives.

    Returns True if the action completed (ACTION_END), False on timeout.
    """
    action = Base_pb2.Action()
    action.name             = "Calibration Waypoint"
    action.application_data = ""

    pose = action.reach_pose.target_pose
    pose.x       = x
    pose.y       = y
    pose.z       = z
    pose.theta_x = theta_x
    pose.theta_y = theta_y
    pose.theta_z = theta_z

    e      = threading.Event()
    handle = base.OnNotificationActionTopic(
        _check_for_end_or_abort(e), Base_pb2.NotificationOptions()
    )
    base.ExecuteAction(action)
    finished = e.wait(timeout_s)
    base.Unsubscribe(handle)
    return finished


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect paired (image, robot pose) data for hand-eye calibration."
    )
    parser.add_argument("--save_dir",   type=str,   default="calibration/extrinsic_data",
                        help="Directory to save images and robot_poses.json")
    parser.add_argument("--n_poses",    type=int,   default=30,
                        help="Number of valid (image, pose) pairs to collect")
    parser.add_argument("--pos_tol",    type=float, default=0.005,
                        help="Position tolerance for waypoint acceptance (m)")
    parser.add_argument("--ori_tol",    type=float, default=1.0,
                        help="Orientation tolerance for waypoint acceptance (deg)")
    # Motion range parameters — tune these for your workspace.
    # NOTE: dx is biased forward (positive X towards the camera) to keep the
    # board facing the scene camera throughout the trajectory.
    parser.add_argument("--dx_min",  type=float, default=0.00,
                        help="Min forward offset from initial pose (m)")
    parser.add_argument("--dx_max",  type=float, default=0.25,
                        help="Max forward offset from initial pose (m)")
    parser.add_argument("--dy_range", type=float, default=0.15,
                        help="Lateral (Y) motion range ±value (m)")
    parser.add_argument("--dz_min",  type=float, default=-0.05,
                        help="Min vertical offset from initial pose (m)")
    parser.add_argument("--dz_max",  type=float, default=0.15,
                        help="Max vertical offset from initial pose (m)")
    parser.add_argument("--drot_xy_range", type=float, default=10.0,
                        help="Pitch/yaw rotation range ±value (deg). "
                             "Keep small to avoid perspective warping of the board.")
    parser.add_argument("--drot_z_range",  type=float, default=15.0,
                        help="Roll rotation range ±value (deg)")
    return parser


def main():
    parser = build_arg_parser()
    args   = parseConnectionArguments(parser=parser)

    os.makedirs(args.save_dir, exist_ok=True)
    pose_log_path = os.path.join(args.save_dir, "robot_poses.json")

    # Resume from a previous partial collection if the log exists.
    pose_data = {}
    if os.path.exists(pose_log_path):
        with open(pose_log_path, "r") as f:
            try:
                pose_data = json.load(f)
            except json.JSONDecodeError:
                print(f"[WARNING] Could not parse {pose_log_path}. Starting fresh.")

    saved_count = len(pose_data)

    print("=" * 55)
    if saved_count >= args.n_poses:
        print(f"[INFO] Already have {saved_count} samples — nothing to do.")
        return
    print(f"[INFO] Resuming. {saved_count}/{args.n_poses} samples collected so far.")
    print("=" * 55)

    # Build ChArUco detector.
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    board      = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y), SQUARE_LENGTH, MARKER_LENGTH, dictionary
    )
    detector = cv2.aruco.CharucoDetector(board)

    with DeviceConnection.createTcpConnection(args) as router:
        base        = BaseClient(router)
        base_cyclic = BaseCyclicClient(router)

        servo_mode = Base_pb2.ServoingModeInformation()
        servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        base.SetServoingMode(servo_mode)

        # Record the initial pose as the exploration reference origin.
        fb = base_cyclic.RefreshFeedback()
        ref_x,  ref_y,  ref_z  = fb.base.tool_pose_x,       fb.base.tool_pose_y,       fb.base.tool_pose_z
        ref_tx, ref_ty, ref_tz = fb.base.tool_pose_theta_x,  fb.base.tool_pose_theta_y, fb.base.tool_pose_theta_z
        print("Reference pose recorded. Ensure the board faces the camera in this pose.")

        camera = RealSenseLatestRGB(width=1280, height=720, fps=30)
        warmup_camera(camera)

        print("=" * 55)
        print("Starting calibration trajectory exploration.")
        print("The arm will move to random waypoints and save frames")
        print("where the ChArUco board is clearly visible.")
        print("=" * 55)

        try:
            while saved_count < args.n_poses:
                # Sample a random offset from the reference pose.
                # X is biased forward to keep the board facing the camera.
                dx  = random.uniform(args.dx_min,        args.dx_max)
                dy  = random.uniform(-args.dy_range,      args.dy_range)
                dz  = random.uniform(args.dz_min,         args.dz_max)
                dtx = random.uniform(-args.drot_xy_range, args.drot_xy_range)
                dty = random.uniform(-args.drot_xy_range, args.drot_xy_range)
                dtz = random.uniform(-args.drot_z_range,  args.drot_z_range)

                target = (
                    ref_x  + dx,  ref_y  + dy,  ref_z  + dz,
                    ref_tx + dtx, ref_ty + dty, ref_tz + dtz,
                )

                print(f"\nWaypoint {saved_count + 1}/{args.n_poses} ...")
                if not move_to_cartesian_pose(base, *target):
                    print("[WARNING] Move timed out. Retrying.")
                    continue

                time.sleep(0.75)  # let the arm settle

                # Verify the robot reached the target accurately.
                fb = base_cyclic.RefreshFeedback()
                actual = (
                    fb.base.tool_pose_x,       fb.base.tool_pose_y,       fb.base.tool_pose_z,
                    fb.base.tool_pose_theta_x, fb.base.tool_pose_theta_y, fb.base.tool_pose_theta_z,
                )
                pos_err = math.sqrt(sum((a - t) ** 2 for a, t in zip(actual[:3], target[:3])))
                ori_err = math.sqrt(sum((a - t) ** 2 for a, t in zip(actual[3:], target[3:])))

                if pos_err > args.pos_tol or ori_err > args.ori_tol:
                    print(
                        f"[REJECTED] Pose not reached accurately "
                        f"(pos: {pos_err * 1000:.1f} mm, ori: {ori_err:.1f}°). Skipping."
                    )
                    continue

                # Check board visibility.
                img, _ = camera.get_latest()
                if img is None:
                    continue

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)

                if charuco_ids is None or len(charuco_ids) < MIN_CHARUCO_CORNERS:
                    print(
                        f"[REJECTED] Board not clearly visible "
                        f"({len(charuco_ids) if charuco_ids is not None else 0} corners). "
                        "Generating new waypoint."
                    )
                    continue

                # Save image and pose.
                filename = f"extrinsic_{saved_count:04d}.png"
                cv2.imwrite(os.path.join(args.save_dir, filename), img)
                pose_data[filename] = {
                    "x": actual[0], "y": actual[1], "z": actual[2],
                    "theta_x": actual[3], "theta_y": actual[4], "theta_z": actual[5],
                }
                with open(pose_log_path, "w") as f:
                    json.dump(pose_data, f, indent=4)

                saved_count += 1
                print(f"[SAVED] {filename} ({len(charuco_ids)} corners).")

        finally:
            print("\nReturning arm to initial pose...")
            move_to_cartesian_pose(base, ref_x, ref_y, ref_z, ref_tx, ref_ty, ref_tz)
            camera.close()

    print("=" * 55)
    print(f"Done. Saved {saved_count} samples to {args.save_dir}/")
    print(f"  Images    : {args.save_dir}/extrinsic_*.png")
    print(f"  Pose log  : {pose_log_path}")
    print("Run run_calibration.py to compute T_BASE_CAMERA.")
    print("=" * 55)


if __name__ == "__main__":
    main()
