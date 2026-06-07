"""
run_calibration.py
------------------
Stage 2 of hand-eye (extrinsic) calibration for the Kinova Gen3 +
Intel RealSense setup.

Reads the paired (image, robot pose) data produced by
collect_calibration_data.py and solves for T_BASE_CAMERA — the 4×4
rigid-body transform from camera frame to robot base frame.

Algorithm
---------
For each image:
  1. Detect ChArUco corners and run SQPnP to get an initial board pose.
  2. Refine with Levenberg-Marquardt (solvePnPRefineLM) for sub-pixel accuracy.

For the full dataset:
  3. Daniilidis dual-quaternion algebraic initialisation (calibrateHandEye).
  4. Dense-graph outlier filtering: drop frames whose T_target_to_gripper
     deviates more than a threshold from the median.
  5. Non-linear SE(3) refinement via scipy.optimize.least_squares (LM),
     minimising the variance of T_target_to_gripper across all retained frames.

The objective is that T_target_to_gripper — the transform of the ChArUco
board relative to the gripper — should be constant across all observations
(it is physically fixed).  Minimising its variance across frames directly
optimises the calibration quality.

Output
------
Prints T_BASE_CAMERA and calibration statistics to stdout.
Copy the printed matrix into camera_config.py as T_BASE_CAMERA.

Run
---
    python run_calibration.py [--data_dir DIR]

See --help for all options.
"""

import argparse
import json
import os

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as SciPyRot


# ---------------------------------------------------------------------------
# ChArUco board parameters
# NOTE: must match the physical board used during data collection.
# ---------------------------------------------------------------------------
SQUARES_X     = 7
SQUARES_Y     = 5
SQUARE_LENGTH = 0.0358   # metres
MARKER_LENGTH = 0.0258   # metres

# Camera intrinsics — must match the camera used during calibration.
# Update these if you recalibrate the camera or use a different unit.
# These are the values used in this project (see camera_config.py).
CAMERA_MATRIX = np.array([
    [927.48284701678767, 0.0,               630.0832023644225 ],
    [0.0,               927.48284701678767, 377.57399658246965],
    [0.0,               0.0,               1.0               ],
], dtype=np.float64)

DISTORTION_COEFFS = np.array([
    0.20970442794012381, -0.65012747660413428, 0.0, 0.0, 0.65864973786044845
], dtype=np.float64)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def homogeneous(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Assemble a 4×4 homogeneous matrix from rotation R and translation t."""
    H = np.eye(4, dtype=np.float64)
    H[:3, :3] = R
    H[:3,  3] = t.flatten()
    return H


def get_kinova_pose(pose_dict: dict, euler_seq: str) -> np.ndarray:
    """Convert a Kinova EE pose dict to a 4×4 T_gripper_to_base matrix.

    Args:
        pose_dict:  Dict with keys x, y, z, theta_x, theta_y, theta_z.
        euler_seq:  Euler angle convention used by the robot firmware
                    (tried as both 'xyz' and 'XYZ' to find the best fit).
    """
    t = np.array([pose_dict["x"], pose_dict["y"], pose_dict["z"]], dtype=np.float64)
    R = SciPyRot.from_euler(
        euler_seq,
        [pose_dict["theta_x"], pose_dict["theta_y"], pose_dict["theta_z"]],
        degrees=True,
    ).as_matrix()
    return homogeneous(R, t)


# ---------------------------------------------------------------------------
# Optimisation
# ---------------------------------------------------------------------------

def objective_function(x, T_g2b_list, T_t2c_list):
    """Residual function for non-linear SE(3) refinement.

    Minimises the variance of T_target_to_gripper across all observations.
    T_target_to_gripper should be constant (the board is rigidly attached
    to the gripper), so minimising its spread directly measures calibration
    quality.

    Args:
        x:           6-DOF parameter vector [rx, ry, rz, tx, ty, tz]
                     encoding T_cam_to_base.
        T_g2b_list:  List of T_gripper_to_base matrices.
        T_t2c_list:  List of T_target_to_cam matrices.

    Returns:
        Concatenated residual vector [translation_residuals, 0.2 * rotation_residuals].
    """
    R_c2b      = SciPyRot.from_rotvec(x[:3]).as_matrix()
    T_cam2base = homogeneous(R_c2b, x[3:6])

    translations, rotvecs = [], []
    for T_g2b, T_t2c in zip(T_g2b_list, T_t2c_list):
        T_t2g = np.linalg.inv(T_g2b) @ T_cam2base @ T_t2c
        translations.append(T_t2g[:3, 3])
        rotvecs.append(SciPyRot.from_matrix(T_t2g[:3, :3]).as_rotvec())

    translations = np.array(translations)
    rotvecs      = np.array(rotvecs)

    err_t = (translations - translations.mean(axis=0)).flatten()
    err_r = (rotvecs      - rotvecs.mean(axis=0)).flatten()

    # Orientation residuals are weighted by ~0.2 to roughly match the
    # translational scale (1 rad ≈ 0.2 m impact at typical distances).
    return np.concatenate([err_t, 0.2 * err_r])


def filter_outliers(
    T_cam2base: np.ndarray,
    T_g2b_list: list,
    T_t2c_list: list,
    threshold_mm: float = 10.0,
):
    """Drop frames whose T_target_to_gripper translation deviates from the median.

    Args:
        T_cam2base:    Current estimate of T_cam_to_base.
        T_g2b_list:    T_gripper_to_base for each frame.
        T_t2c_list:    T_target_to_cam for each frame.
        threshold_mm:  Max allowed deviation from median (mm).

    Returns:
        (valid_indices, per_frame_errors_mm)
    """
    estimates = []
    for T_g2b, T_t2c in zip(T_g2b_list, T_t2c_list):
        T_t2g = np.linalg.inv(T_g2b) @ T_cam2base @ T_t2c
        estimates.append(T_t2g[:3, 3])

    estimates  = np.array(estimates)
    median     = np.median(estimates, axis=0)
    errors_mm  = np.linalg.norm(estimates - median, axis=1) * 1000.0
    valid_idx  = [i for i, e in enumerate(errors_mm) if e <= threshold_mm]
    return valid_idx, errors_mm


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute T_BASE_CAMERA from ChArUco hand-eye calibration data."
    )
    parser.add_argument("--data_dir",       type=str,   default="calibration/extrinsic_data",
                        help="Directory containing images and robot_poses.json")
    parser.add_argument("--outlier_thresh", type=float, default=12.0,
                        help="Outlier rejection threshold after initial solve (mm)")
    args = parser.parse_args()

    pose_log = os.path.join(args.data_dir, "robot_poses.json")
    if not os.path.exists(pose_log):
        print(f"[ERROR] Pose log not found: {pose_log}")
        return

    with open(pose_log, "r") as f:
        pose_data = json.load(f)

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    board      = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y), SQUARE_LENGTH, MARKER_LENGTH, dictionary
    )
    detector = cv2.aruco.CharucoDetector(board)

    print(f"Loaded {len(pose_data)} samples.  Estimating board poses...")

    image_names    = []
    T_t2c_list     = []

    for filename in pose_data:
        img_path = os.path.join(args.data_dir, filename)
        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        charuco_corners, charuco_ids, _, _ = detector.detectBoard(img)

        if charuco_ids is None or len(charuco_ids) < 12:
            continue

        obj_pts, img_pts = board.matchImagePoints(charuco_corners, charuco_ids)
        if obj_pts is None or len(obj_pts) < 6:
            continue

        # Step 1: SQPnP (globally optimal for planar scenes).
        ok, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts, CAMERA_MATRIX, DISTORTION_COEFFS,
            flags=cv2.SOLVEPNP_SQPNP,
        )
        if not ok:
            continue

        # Step 2: LM sub-pixel refinement.
        rvec, tvec = cv2.solvePnPRefineLM(
            obj_pts, img_pts, CAMERA_MATRIX, DISTORTION_COEFFS, rvec, tvec
        )
        R_t2c, _ = cv2.Rodrigues(rvec)
        T_t2c_list.append(homogeneous(R_t2c, tvec))
        image_names.append(filename)

    if len(T_t2c_list) < 5:
        print(f"[ERROR] Only {len(T_t2c_list)} valid frames — need at least 5.")
        return

    print(f"Valid PnP observations: {len(T_t2c_list)}")
    print("\nRunning optimisation...")

    # Try both Euler conventions used by Kinova firmware.
    # Kinova Gen3 uses intrinsic XYZ (≡ extrinsic ZYX); 'XYZ' is the
    # scipy convention for intrinsic rotations.
    candidate_seqs = ["xyz", "XYZ"]

    best        = None
    best_error  = float("inf")

    for seq in candidate_seqs:
        T_g2b_list = [get_kinova_pose(pose_data[fn], seq) for fn in image_names]

        # Step 3: Daniilidis dual-quaternion algebraic initialisation.
        R_b2g_list = [np.linalg.inv(T)[:3, :3]   for T in T_g2b_list]
        t_b2g_list = [np.linalg.inv(T)[:3, 3:4]  for T in T_g2b_list]
        R_t2c_raw  = [T[:3, :3]                   for T in T_t2c_list]
        t_t2c_raw  = [T[:3, 3:4]                  for T in T_t2c_list]

        R_c2b_init, t_c2b_init = cv2.calibrateHandEye(
            R_gripper2base=R_b2g_list, t_gripper2base=t_b2g_list,
            R_target2cam=R_t2c_raw,   t_target2cam=t_t2c_raw,
            method=cv2.CALIB_HAND_EYE_DANIILIDIS,
        )
        T_c2b_init = homogeneous(R_c2b_init, t_c2b_init)

        # Step 4: Outlier filtering.
        valid_idx, _ = filter_outliers(
            T_c2b_init, T_g2b_list, T_t2c_list,
            threshold_mm=args.outlier_thresh,
        )
        if len(valid_idx) < 5:
            print(f"  [{seq}] Too few inliers after filtering ({len(valid_idx)}). Skipping.")
            continue

        T_g2b_in = [T_g2b_list[i] for i in valid_idx]
        T_t2c_in = [T_t2c_list[i] for i in valid_idx]

        # Step 5: Non-linear SE(3) refinement.
        x0 = np.concatenate([
            SciPyRot.from_matrix(R_c2b_init).as_rotvec(),
            t_c2b_init.flatten(),
        ])
        res = least_squares(
            objective_function, x0,
            args=(T_g2b_in, T_t2c_in),
            method="lm",
            xtol=1e-8, ftol=1e-8, max_nfev=2000,
        )
        T_opt = homogeneous(SciPyRot.from_rotvec(res.x[:3]).as_matrix(), res.x[3:6])

        _, final_errs = filter_outliers(T_opt, T_g2b_in, T_t2c_in, threshold_mm=100.0)
        mean_err = float(np.mean(final_errs))

        print(f"  [{seq}]  inliers={len(valid_idx)}  mean_err={mean_err:.2f} mm")

        if mean_err < best_error:
            best_error = mean_err
            best = {
                "seq":      seq,
                "T":        T_opt,
                "inliers":  len(valid_idx),
                "mean_mm":  mean_err,
                "std_mm":   float(np.std(final_errs)),
                "max_mm":   float(np.max(final_errs)),
            }

    if best is None:
        print("\n[ERROR] Calibration failed. Check data quality.")
        return

    T = best["T"]
    np.set_printoptions(precision=5, suppress=True)

    print("\n" + "=" * 60)
    print("HAND-EYE CALIBRATION COMPLETE")
    print("=" * 60)
    print("T_BASE_CAMERA  (T_cam_to_base):")
    print("-" * 60)
    print(T)
    print("\n" + "=" * 60)
    print("STATISTICS")
    print("-" * 60)
    print(f"Euler sequence  : {best['seq']}")
    print(f"Inlier frames   : {best['inliers']}")
    print(f"Mean drift      : {best['mean_mm']:.3f} mm")
    print(f"Std dev         : {best['std_mm']:.3f} mm")
    print(f"Max error       : {best['max_mm']:.3f} mm")
    cam_pos = T[:3, 3]
    print(f"\nCamera position in base frame:")
    print(f"  X={cam_pos[0]:.4f} m  Y={cam_pos[1]:.4f} m  Z={cam_pos[2]:.4f} m")
    print("\nCopy the matrix above into camera_config.py as T_BASE_CAMERA.")
    print("=" * 60)


if __name__ == "__main__":
    main()
