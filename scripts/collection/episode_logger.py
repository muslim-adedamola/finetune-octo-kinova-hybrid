"""
episode_logger.py
-----------------
Core logging infrastructure for Kinova Gen3 data collection.

Provides:
  EpisodeLogger    — writes per-timestep robot state and RGB images to disk
                     in the episode format expected by the TFDS dataset builder.
  RealSenseLatestRGB — threaded Intel RealSense RGB capture.
  create_arm_camera_grabber — starts an ffmpeg RTSP reader for the wrist camera.

Episode output layout
---------------------
<out_root>/
  episode_<YYYYMMDD_HHMMSS>/
    episode.csv          # one row per timestep (see EpisodeLogger._fieldnames)
    metadata.json        # episode-level metadata and success flag
    rgb/                 # scene camera frames  (<step_idx:06d>.png)
    arm_camera/          # wrist camera frames  (<step_idx:06d>.png)  [optional]

CSV columns
-----------
  t_wall, dt_wall                      — wall-clock time and step duration (s)
  t_img, img_file                      — scene camera timestamp and filename
  tool_pose_{x,y,z,theta_x,y,z}       — measured EE pose (m / deg)
  cmd_tool_pose_{x,y,z,theta_x,y,z}   — commanded EE pose (m / deg)
  a_d{x,y,z,theta_x,y,z}              — delta-action (commanded[t] − commanded[t-1])
  joint_{1..7}_deg                     — joint angles (deg)
  gripper_pos_norm                     — normalized gripper in [0, 1]  (1=open, 0=closed)
  gripper_state_bin                    — binary gripper state (1=open, 0=closed)
  a_dgripper                           — gripper delta action
"""

import csv
import json
import os
import signal
import subprocess
import threading
import time
from datetime import datetime

import cv2
import numpy as np
import pyrealsense2 as rs

from arm_frame_grabber import FrameGrabber
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient

# Import from the shared utilities module already present in scripts/deployment/.
# When running from scripts/collection/, add scripts/deployment/ to PYTHONPATH or
# adjust this import to a relative path if you restructure the project.
from utilities import DeviceConnection, parseConnectionArguments


# ---------------------------------------------------------------------------
# Gripper normalization
# NOTE: OPEN_RAW and CLOSED_RAW are raw motor-position readings from the
# Kinova gripper at fully open and fully closed states. These values were
# measured on the specific robot unit used in this project.
# Run the robot manually and read fb.interconnect.gripper_feedback.motor[0].position
# to calibrate for your own unit.
# ---------------------------------------------------------------------------
OPEN_RAW   = 0.8733651041984558
CLOSED_RAW = 99.12664031982422


def normalize_gripper(raw: float) -> float:
    """Convert raw motor position to a normalized value in [0, 1].

    Returns 1.0 for fully open, 0.0 for fully closed.
    """
    g = (raw - OPEN_RAW) / (CLOSED_RAW - OPEN_RAW)
    g = max(0.0, min(1.0, g))
    return 1.0 - g


def binarize_gripper(gr_norm: float, prev_state=None) -> int:
    """Convert normalized gripper to a binary open/close state with hysteresis.

    Thresholds:
      gr_norm <= 0.6  → 0 (closed)
      gr_norm >= 0.7  → 1 (open)
      0.6 < gr_norm < 0.7 → keep previous state (hysteresis band)

    Args:
        gr_norm:    Normalized gripper value in [0, 1].
        prev_state: Previous binary state (0 or 1), used in the hysteresis band.

    Returns:
        0 (closed) or 1 (open).
    """
    if gr_norm <= 0.6:
        return 0
    if gr_norm >= 0.7:
        return 1
    return prev_state if prev_state is not None else 0


def wrap_deg(d: float) -> float:
    """Wrap an angle difference into [-180, 180)."""
    return (d + 180.0) % 360.0 - 180.0


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------------
# RTSP wrist-camera grabber
# ---------------------------------------------------------------------------

def create_arm_camera_grabber(
    url: str = "rtsp://192.168.1.13/color",  # NOTE: replace with your robot's IP
    startup_timeout_s: float = 5.0,
) -> FrameGrabber:
    """Start an ffmpeg RTSP reader and return a FrameGrabber.

    Probes the stream for its resolution, then launches ffmpeg to decode
    frames into a raw BGR24 pipe consumed by FrameGrabber.

    Args:
        url:               RTSP URL of the arm-mounted camera stream.
        startup_timeout_s: Seconds to wait for the first frame before raising.

    Raises:
        RuntimeError: If no frames are received within *startup_timeout_s*.
    """
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "json", url,
    ]
    probe = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
    info = json.loads(probe.stdout)
    width  = int(info["streams"][0]["width"])
    height = int(info["streams"][0]["height"])
    print(f"Arm camera stream resolution: {width}×{height}")

    ffmpeg_cmd = [
        "ffmpeg",
        "-rtsp_transport", "tcp",
        "-flags", "low_delay",
        "-fflags", "nobuffer",
        "-i", url,
        "-an",
        "-c:v", "rawvideo",
        "-pix_fmt", "bgr24",
        "-f", "rawvideo", "pipe:1",
    ]
    proc = subprocess.Popen(
        ffmpeg_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=10**8,
    )
    grabber = FrameGrabber(proc, width, height)

    t0 = time.time()
    while time.time() - t0 < startup_timeout_s:
        if grabber.get_latest_frame() is not None:
            print("Arm camera stream ready.")
            return grabber
        time.sleep(0.05)

    # Startup failed — clean up before raising.
    try:
        proc.terminate()
        proc.wait(timeout=1.0)
    except Exception:
        pass
    grabber.stop()
    raise RuntimeError(
        f"Arm camera stream at {url!r} started but produced no frames "
        f"within {startup_timeout_s:.1f}s."
    )


# ---------------------------------------------------------------------------
# Keyboard listener (teleop / manual-recording mode)
# ---------------------------------------------------------------------------

def _getch():
    """Read one character from stdin without pressing Enter (best-effort).
    Works on Linux/macOS; falls back to msvcrt on Windows.
    """
    try:
        import sys
        import tty
        import termios
        fd = sys.stdin.fileno()
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


def keyboard_listener(logger: "EpisodeLogger"):
    """Background thread: watch for s / f / q keypresses.

    s → mark episode SUCCESS and stop.
    f → mark episode FAIL and stop.
    q → stop without setting a success flag.
    """
    while not logger._stop:
        ch = _getch()
        if not ch:
            continue
        ch = ch.lower()
        if ch == "s":
            logger.success = True
            logger.request_stop()
        elif ch == "f":
            logger.success = False
            logger.request_stop()
        elif ch == "q":
            logger.success = None
            logger.request_stop()


# ---------------------------------------------------------------------------
# Episode logger
# ---------------------------------------------------------------------------

# Columns that must be present for the TFDS dataset builder and replay scripts.
REPLAY_REQUIRED_COLUMNS = [
    "dt_wall",
    "cmd_tool_pose_x", "cmd_tool_pose_y", "cmd_tool_pose_z",
    "cmd_tool_pose_theta_x", "cmd_tool_pose_theta_y", "cmd_tool_pose_theta_z",
    "a_dx", "a_dy", "a_dz",
    "a_dtheta_x", "a_dtheta_y", "a_dtheta_z",
]


class EpisodeLogger:
    """Writes per-timestep robot state and camera images for one episode.

    Creates a timestamped episode directory under *out_root* and opens a
    CSV file immediately. Call log_step() at each control tick and close()
    (or use as a context manager) when the episode ends.

    Args:
        out_root:        Root directory for all episodes.
        hz:              Target control / logging frequency (Hz).
        num_joints:      Number of arm joints to record (default 7 for Gen3).
        save_arm_camera: Whether to save arm/wrist camera frames alongside
                         the scene camera.
    """

    def __init__(
        self,
        out_root: str,
        hz: float,
        num_joints: int = 7,
        save_arm_camera: bool = True,
    ):
        self.out_root = out_root
        self.hz = hz
        self.num_joints = num_joints
        self.dt_target = 1.0 / hz
        self._stop = False
        self._prev_gr_bin = None
        self.save_arm_camera = save_arm_camera

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.ep_dir = os.path.join(out_root, f"episode_{ts}")
        _ensure_dir(self.ep_dir)

        self.rgb_dir = os.path.join(self.ep_dir, "rgb")
        _ensure_dir(self.rgb_dir)

        self.arm_camera_dir = os.path.join(self.ep_dir, "arm_camera")
        if self.save_arm_camera:
            _ensure_dir(self.arm_camera_dir)

        self.step_idx = 0
        self.csv_path  = os.path.join(self.ep_dir, "episode.csv")
        self.meta_path = os.path.join(self.ep_dir, "metadata.json")

        joint_headers = [f"joint_{i + 1}_deg" for i in range(self.num_joints)]
        self._fieldnames = [
            "t_wall", "dt_wall",
            "t_img", "img_file",
            # measured EE pose
            "tool_pose_x", "tool_pose_y", "tool_pose_z",
            "tool_pose_theta_x", "tool_pose_theta_y", "tool_pose_theta_z",
            # commanded EE pose
            "cmd_tool_pose_x", "cmd_tool_pose_y", "cmd_tool_pose_z",
            "cmd_tool_pose_theta_x", "cmd_tool_pose_theta_y", "cmd_tool_pose_theta_z",
            # delta actions (commanded[t] - commanded[t-1])
            "a_dx", "a_dy", "a_dz",
            "a_dtheta_x", "a_dtheta_y", "a_dtheta_z",
            # joint angles
            *joint_headers,
            # gripper
            "gripper_pos_norm", "gripper_state_bin", "a_dgripper",
        ]

        missing = [c for c in REPLAY_REQUIRED_COLUMNS if c not in self._fieldnames]
        if missing:
            raise RuntimeError(f"Logger schema is missing required replay columns: {missing}")

        self._f = open(self.csv_path, "w", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=self._fieldnames)
        self._w.writeheader()
        self._f.flush()

        self._t_prev   = None
        self._prev_cmd = None   # {x, y, z, tx, ty, tz}
        self._prev_gr  = None
        self.success   = None   # set before close(): True = success, False = fail

        meta = {
            "episode_dir":      os.path.basename(self.ep_dir),
            "created_at":       ts,
            "logger_hz_target": self.hz,
            "save_arm_camera":  self.save_arm_camera,
            "notes":            "Raw BaseCyclic feedback. Joint angles in degrees.",
        }
        with open(self.meta_path, "w") as mf:
            json.dump(meta, mf, indent=2)

    def request_stop(self):
        self._stop = True

    def close(self):
        """Flush and close the CSV, then write the final success flag to metadata."""
        if not self._f.closed:
            self._f.flush()
            self._f.close()

        with open(self.meta_path, "r") as mf:
            meta = json.load(mf)
        meta["success"] = self.success
        with open(self.meta_path, "w") as mf:
            json.dump(meta, mf, indent=2)

    def log_step(self, b, gr, joint_angles=None, img=None, t_img=None, arm_img=None):
        """Record one timestep.

        Args:
            b:            BaseCyclic base feedback object.
            gr:           Normalized gripper value in [0, 1].
            joint_angles: List of joint angles in degrees (length == num_joints).
            img:          Scene camera frame (BGR numpy array), or None.
            t_img:        Timestamp of *img*, or None.
            arm_img:      Wrist camera frame (BGR numpy array), or None.
        """
        t = time.time()
        dt_wall = 0.0 if self._t_prev is None else (t - self._t_prev)
        self._t_prev = t

        # Save scene camera frame.
        img_file = ""
        if img is not None:
            img_file = f"{self.step_idx:06d}.png"
            cv2.imwrite(os.path.join(self.rgb_dir, img_file), img)

        # Save wrist camera frame.
        if self.save_arm_camera and arm_img is not None:
            cv2.imwrite(
                os.path.join(self.arm_camera_dir, f"{self.step_idx:06d}.png"),
                arm_img,
            )

        # Measured EE pose.
        mx, my, mz = b.tool_pose_x, b.tool_pose_y, b.tool_pose_z
        mtx, mty, mtz = (
            b.tool_pose_theta_x,
            b.tool_pose_theta_y,
            b.tool_pose_theta_z,
        )

        # Commanded EE pose.
        cx, cy, cz = (
            b.commanded_tool_pose_x,
            b.commanded_tool_pose_y,
            b.commanded_tool_pose_z,
        )
        ctx, cty, ctz = (
            b.commanded_tool_pose_theta_x,
            b.commanded_tool_pose_theta_y,
            b.commanded_tool_pose_theta_z,
        )

        # Delta action (commanded[t] - commanded[t-1]).
        if self._prev_cmd is None:
            a_dx = a_dy = a_dz = a_dtx = a_dty = a_dtz = 0.0
        else:
            a_dx  = cx  - self._prev_cmd["x"]
            a_dy  = cy  - self._prev_cmd["y"]
            a_dz  = cz  - self._prev_cmd["z"]
            a_dtx = wrap_deg(ctx - self._prev_cmd["tx"])
            a_dty = wrap_deg(cty - self._prev_cmd["ty"])
            a_dtz = wrap_deg(ctz - self._prev_cmd["tz"])

        # Binary gripper action.
        gr_bin = binarize_gripper(gr, self._prev_gr_bin)
        a_dgr  = 0 if self._prev_gr_bin is None else gr_bin - self._prev_gr_bin

        self._prev_gr_bin = gr_bin
        self._prev_cmd    = {"x": cx, "y": cy, "z": cz, "tx": ctx, "ty": cty, "tz": ctz}
        self._prev_gr     = gr

        # Pad or truncate joint angles to match the header width.
        if joint_angles is None:
            joint_angles = []
        joint_row = list(joint_angles[: self.num_joints])
        joint_row.extend([""] * (self.num_joints - len(joint_row)))

        row = {
            "t_wall":   t,
            "dt_wall":  dt_wall,
            "t_img":    t_img if t_img is not None else "",
            "img_file": img_file,
            "tool_pose_x": mx, "tool_pose_y": my, "tool_pose_z": mz,
            "tool_pose_theta_x": mtx, "tool_pose_theta_y": mty, "tool_pose_theta_z": mtz,
            "cmd_tool_pose_x": cx, "cmd_tool_pose_y": cy, "cmd_tool_pose_z": cz,
            "cmd_tool_pose_theta_x": ctx, "cmd_tool_pose_theta_y": cty, "cmd_tool_pose_theta_z": ctz,
            "a_dx": a_dx, "a_dy": a_dy, "a_dz": a_dz,
            "a_dtheta_x": a_dtx, "a_dtheta_y": a_dty, "a_dtheta_z": a_dtz,
            "gripper_pos_norm": gr, "gripper_state_bin": gr_bin, "a_dgripper": a_dgr,
        }
        for i in range(self.num_joints):
            row[f"joint_{i + 1}_deg"] = joint_row[i]

        self._w.writerow(row)
        self.step_idx += 1

    def sleep_to_rate(self, loop_start_t: float):
        """Sleep for the remainder of the current control tick."""
        remaining = self.dt_target - (time.time() - loop_start_t)
        if remaining > 0:
            time.sleep(remaining)


# ---------------------------------------------------------------------------
# Threaded RealSense scene camera
# ---------------------------------------------------------------------------

class RealSenseLatestRGB:
    """Captures BGR frames from an Intel RealSense camera in a background thread.

    Calling get_latest() always returns the most recent frame without blocking
    the control loop.

    Args:
        width, height: Resolution of the color stream.
        fps:           Frame rate.
    """

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        self.width  = width
        self.height = height
        self.fps    = fps

        pipeline = rs.pipeline()
        config   = rs.config()
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        pipeline.start(config)
        self.pipeline = pipeline

        self._lock       = threading.Lock()
        self._latest_img = None
        self._latest_t   = None
        self._stop       = False

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop:
            try:
                frames = self.pipeline.wait_for_frames()
                color  = frames.get_color_frame()
                if not color:
                    continue
                img = np.asanyarray(color.get_data())
                t   = time.time()
                with self._lock:
                    self._latest_img = img
                    self._latest_t   = t
            except Exception:
                continue  # don't let a camera hiccup kill the logger

    def get_latest(self):
        """Return (image, timestamp) for the most recent frame, or (None, None)."""
        with self._lock:
            if self._latest_img is None:
                return None, None
            return self._latest_img.copy(), self._latest_t

    def close(self):
        self._stop = True
        try:
            self._thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            self.pipeline.stop()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Standalone entry point — manual / teleoperation recording
# ---------------------------------------------------------------------------

def main():
    """Record a single episode while a human teleoperates the robot.

    Press:
      s  → stop and mark the episode as SUCCESS
      f  → stop and mark the episode as FAIL
      q  → stop without a success label
      Ctrl+C also stops.
    """
    parser = parseConnectionArguments.__wrapped__ if hasattr(parseConnectionArguments, "__wrapped__") else None

    import argparse
    arg_parser = argparse.ArgumentParser(description="Manual teleop episode recorder")
    arg_parser.add_argument("--out_root",   type=str,   default="kinova_raw", help="Output root directory")
    arg_parser.add_argument("--hz",         type=float, default=20.0,         help="Logging frequency (Hz)")
    arg_parser.add_argument("--skip_frames",type=int,   default=30,           help="Initial frames to skip while stream stabilizes")
    arg_parser.add_argument("--no_arm_camera", action="store_true",           help="Disable wrist camera capture")
    args = parseConnectionArguments(parser=arg_parser)

    print("\nTeleop Episode Recorder")
    print(f"  Output : {args.out_root}/")
    print(f"  Rate   : {args.hz} Hz")
    print("\nControls:")
    print("  s  → SUCCESS")
    print("  f  → FAIL")
    print("  q  → quit (no label)")
    print("  Ctrl+C also stops\n")

    _ensure_dir(args.out_root)

    logger     = EpisodeLogger(out_root=args.out_root, hz=args.hz)
    camera     = RealSenseLatestRGB(width=640, height=480, fps=30)
    arm_grabber = None
    if not args.no_arm_camera:
        try:
            arm_grabber = create_arm_camera_grabber()
        except Exception as exc:
            print(f"Warning: could not start arm camera: {exc}")

    print("Warming up camera...")
    time.sleep(0.5)
    for _ in range(30):
        img, _ = camera.get_latest()
        if img is not None:
            break
        time.sleep(0.03)
    print("Camera ready.")

    def _handle_sigint(sig, frame):
        logger.request_stop()
    signal.signal(signal.SIGINT, _handle_sigint)

    threading.Thread(target=keyboard_listener, args=(logger,), daemon=True).start()

    with DeviceConnection.createUdpConnection(args) as router:
        base_cyclic = BaseCyclicClient(router)
        skip_left   = args.skip_frames

        try:
            while not logger._stop:
                loop_t0 = time.time()
                fb      = base_cyclic.RefreshFeedback()
                b       = fb.base
                gr      = normalize_gripper(fb.interconnect.gripper_feedback.motor[0].position)
                joints  = [act.position for act in fb.actuators]
                img, t_img = camera.get_latest()

                if img is None:
                    logger.sleep_to_rate(loop_t0)
                    continue

                arm_img = None
                if arm_grabber is not None:
                    arm_img = arm_grabber.get_latest_frame()

                if skip_left > 0:
                    skip_left -= 1
                    logger.sleep_to_rate(loop_t0)
                    continue

                logger.log_step(b, gr, joint_angles=joints, img=img, t_img=t_img, arm_img=arm_img)

                # Periodic flush to reduce data loss risk.
                if int(loop_t0 * 2) != int((loop_t0 - logger.dt_target) * 2):
                    logger._f.flush()

                logger.sleep_to_rate(loop_t0)

        finally:
            logger.close()
            camera.close()
            if arm_grabber is not None:
                try:
                    arm_grabber.proc.terminate()
                    arm_grabber.proc.wait(timeout=1.0)
                except Exception:
                    pass
                arm_grabber.stop()

    print(f"\nEpisode saved to : {logger.ep_dir}")
    print(f"  CSV            : {logger.csv_path}")
    print(f"  Metadata       : {logger.meta_path}")


if __name__ == "__main__":
    main()
