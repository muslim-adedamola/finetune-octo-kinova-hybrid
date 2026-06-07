# Data Collection

A description of how to use the scripts in `scripts/collection/` to collect Kinova Gen3 bottle pick-and-lift demonstrations in the episode format expected by the TFDS dataset builder.

---

## Overview

Collection is **scripted and automated** — no teleoperation device is needed.  For each episode:

1. The scene camera detects the bottle with YOLOE and projects its centre to world coordinates.
2. The robot autonomously executes: Home → approach → grasp → lift.
3. The episode is logged to disk and is ready to be fed into the TFDS builder.

A separate teleop path (`episode_logger.py --help`) is also available if you want to collect demonstrations by manually guiding the robot.

---

## Hardware Requirements

| Component | Details used in this project |
|---|---|
| Robot arm | Kinova Gen3, 7 DoF |
| Scene camera | Intel RealSense D435 (external, third-person view) |
| Wrist camera | Kinova arm camera, accessed via RTSP |
| Network | Robot reachable at `192.168.1.13` (default Kinova IP) |

---

## Software Requirements

Install all dependencies listed in `requirements.txt`, plus:

- Kinova Kortex Python API
- Intel RealSense SDK and `pyrealsense2`
- `ultralytics` (YOLOE)
- `ffmpeg` on the system PATH (for wrist camera RTSP decoding)

---

## File Structure

```
scripts/collection/
  camera_config.py        — camera intrinsics, extrinsics, object-plane constants
  arm_frame_grabber.py    — threaded RTSP frame reader for the wrist camera
  episode_logger.py       — EpisodeLogger, RealSenseLatestRGB, arm camera grabber
  trajectory_primitives.py — PD controller, gripper commands, motion primitives
  collect_episode.py      — main entry point: detect → plan → execute → log
  detect_bottle_live.py   — diagnostic: live detection + world-coordinate overlay

scripts/deployment/
  utilities.py            — Kortex connection helpers (shared with collection scripts)
```

---

## Calibration

Before collecting data you must update `camera_config.py` with values for your own rig.

### Camera intrinsics

Run the intrinsics camera calibration to get the intrinsics values.

```bash
rs-enumerate-devices -c
```

Update `CAMERA_MATRIX` and `DIST_COEFFS` in `camera_config.py`.

### Camera extrinsics (`T_BASE_CAMERA`)

`T_BASE_CAMERA` is the 4×4 rigid-body transform from camera frame to robot base frame.  Obtain it via hand-eye calibration, for example using [easy_handeye](https://github.com/IFL-CAMP/easy_handeye) or by recording robot EE poses alongside ArUco marker detections.

### Gripper normalization (`OPEN_RAW`, `CLOSED_RAW`)

These are raw motor-position readings from the Kinova gripper at fully open and fully closed states.  To measure them, run the robot manually and print:

```python
fb.interconnect.gripper_feedback.motor[0].position
```

at each extreme, then update the constants in `episode_logger.py`.

### Object height (`BOTTLE_Z_WORLD`)

Measure the height of your target object's centre above the table surface in metres and update `BOTTLE_Z_WORLD` in `camera_config.py`.

### Detection shift (`SHIFT_VEC_BOTTLES`)

This is an empirical XY offset applied after projecting the detection centroid to world coordinates.  Use `detect_bottle_live.py` to verify that the displayed world coordinates match the physical bottle position, then tune `SHIFT_VEC_BOTTLES` until they agree.

---

## YOLOE Reference Image and Visual Prompt

Both `collect_episode.py` and `detect_bottle_live.py` use YOLOE in visual-prompt mode:

1. Capture a clear image of your target object (`sample_image.png` in this project).
2. Note the pixel bounding box around the object in that image.
3. Update `REFERENCE_IMAGE_PATH` and `VISUAL_PROMPTS` at the top of each script.

---

## Running the Diagnostic

Before collecting real episodes, verify detection and projection with the live diagnostic:

```bash
cd scripts/collection
python detect_bottle_live.py
```

The window shows the live camera feed with the detected bounding box and projected world coordinates overlaid.  Press **Q** to quit.

Confirm that:
- The bottle is consistently detected.
- The displayed X/Y/Z world coordinates match the physical bottle position.

---

## Collecting Episodes

```bash
cd scripts/collection
python collect_episode.py \
  --out_root /path/to/kinova_bottle_lift_raw \
  --hz 10 \
  --lift_height 0.20 \
  --hold_time 2.0
```

**ESC** at any point aborts the motion, returns the arm to Home, and deletes the episode directory.  Only completed episodes are saved.

Repeat for as many episodes as needed.  Each episode is saved to a new timestamped directory:

```
kinova_bottle_lift_raw/
  episode_20240601_103012/
    episode.csv
    metadata.json
    rgb/
      000000.png
      000001.png
      ...
    arm_camera/
      000000.png
      ...
```

### Key options

| Flag | Default | Description |
|---|---|---|
| `--out_root` | `kinova_bottle_lift_raw` | Root directory for episodes |
| `--hz` | `10.0` | Control and logging frequency |
| `--lift_height` | `0.20` | Height above bottle Z for lift pose (m) |
| `--hold_time` | `2.0` | Seconds to hold bottle after lift |
| `--detect_timeout` | `15.0` | Abort if no bottle detected within this time (s) |
| `--no_arm_camera` | off | Disable wrist camera capture |
| `--no_log_images` | off | Disable image saving entirely |

Run `python collect_episode.py --help` for the full list including PD gains.

---

## Episode Output Format

Each episode directory matches the format expected by the TFDS dataset builder in `scripts/dataset/`.  See [docs/DATASET.md](DATASET.md) for details on converting raw episodes to TFDS/RLDS format.

### CSV columns

| Column | Description |
|---|---|
| `t_wall`, `dt_wall` | Wall-clock timestamp and step duration (s) |
| `tool_pose_{x,y,z,theta_x,y,z}` | Measured EE pose (m / deg) |
| `cmd_tool_pose_{x,y,z,theta_x,y,z}` | Commanded EE pose (m / deg) |
| `a_d{x,y,z,theta_x,y,z}` | Delta action: commanded[t] − commanded[t−1] |
| `joint_{1..7}_deg` | Joint angles (deg) |
| `gripper_pos_norm` | Normalized gripper in [0, 1] (1 = open, 0 = closed) |
| `gripper_state_bin` | Binary gripper state (1 = open, 0 = closed) |
| `a_dgripper` | Gripper delta action |

---

## Manual Teleop Recording

If you prefer to collect demonstrations by manually guiding the robot (kinesthetic teaching with an external controller), use the standalone teleop recorder:

```bash
cd scripts/collection
python episode_logger.py --out_root /path/to/kinova_raw --hz 20
```

| Key | Action |
|---|---|
| `s` | Mark episode SUCCESS and stop |
| `f` | Mark episode FAIL and stop |
| `q` | Stop without a success label |
| Ctrl+C | Stop |

Only episodes marked with `s` are used by the dataset builder.
