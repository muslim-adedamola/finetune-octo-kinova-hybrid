"""
camera_config.py
----------------
Camera intrinsics, extrinsics, and object-plane constants for the
Kinova Gen3 + Intel RealSense D435 setup used during data collection.

All values are specific to the physical rig used in this project.
If you are adapting this pipeline to a different robot or camera
mounting, you must replace:

  - CAMERA_MATRIX      : run RealSense camera calibration or use
                         rs-enumerate-devices / realsense-viewer
  - DIST_COEFFS        : same calibration procedure
  - T_BASE_CAMERA      : run hand-eye calibration (e.g. easy_handeye,
                         or record robot EE poses + ArUco detections)
  - SHIFT_VEC_BOTTLES  : empirical offset tuned for your camera-to-object
                         depth and detection centroid bias
  - BOTTLE_Z_WORLD     : measure the height of the target object's
                         centre above the table surface in metres
"""

import numpy as np

# ---------------------------------------------------------------------------
# Camera intrinsics
# NOTE: these were obtained from the RealSense SDK for the specific unit
# used in this project. Replace with values from your own camera.
# can also be obtained using chaARuco marker board for intrinsics calibration.
# ---------------------------------------------------------------------------
CAMERA_MATRIX = np.array([
    [927.48284701678767, 0.0,                630.0832023644225],
    [0.0,               927.48284701678767,  377.57399658246965],
    [0.0,               0.0,                 1.0],
], dtype=np.float64)

DIST_COEFFS = np.array([
    0.20970442794012381,
    -0.65012747660413428,
    0.0,
    0.0,
    0.65864973786044845,
], dtype=np.float64)

# ---------------------------------------------------------------------------
# Camera extrinsics — transform from camera frame to robot base frame
# T_BASE_CAMERA is a 4×4 rigid-body transformation matrix:
#   p_base = T_BASE_CAMERA @ p_camera
# NOTE: obtained via the hand-eye calibration pipeline in scripts/calibration/.
# To recalibrate for your own setup:
#   1. python scripts/calibration/collect_calibration_data.py
#   2. python scripts/calibration/run_calibration.py
# Copy the printed T_BASE_CAMERA matrix here.
# ---------------------------------------------------------------------------
T_BASE_CAMERA = np.array(
    [[ 0.21883,  0.54539, -0.80911,  1.71418],
     [ 0.97494, -0.08806,  0.20432,  0.07002],
     [ 0.04018, -0.83354, -0.55099,  1.04878],
     [ 0.00000,  0.00000,  0.00000,  1.00000]],
    dtype=np.float64,
)

# ---------------------------------------------------------------------------
# Object-plane constants
# ---------------------------------------------------------------------------

# World-frame Z height of the bottle centre (metres above table surface).
# NOTE: measure this for your own object and table setup.
BOTTLE_Z_WORLD: float = 0.06

# Empirical XY shift applied after projecting the detection centroid to
# world coordinates, to correct for camera-to-object offset and centroid
# bias from the YOLOE bounding box.
# NOTE: tune this for your camera mounting and object geometry.
SHIFT_VEC_BOTTLES = np.array([-0.1, 0.015, 0.0], dtype=np.float64)
