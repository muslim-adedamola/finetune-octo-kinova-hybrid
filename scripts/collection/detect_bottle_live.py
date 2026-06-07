"""
detect_bottle_live.py
---------------------
Live diagnostic tool: runs YOLOE bottle detection on the RealSense stream
and overlays the detected bounding box and projected world coordinates on
the camera feed.

Use this script to:
  - Verify the YOLOE model detects your target object reliably.
  - Confirm that the camera-to-world projection (intrinsics + extrinsics)
    is calibrated correctly before collecting episodes.
  - Tune SHIFT_VEC_BOTTLES in camera_config.py.

Press Q to quit.

Usage
-----
    python detect_bottle_live.py

No robot connection is required — the script only uses the camera.
"""

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
from episode_logger import RealSenseLatestRGB


# ---------------------------------------------------------------------------
# YOLOE visual prompt
# NOTE: Replace REFERENCE_IMAGE_PATH with your own reference image and
# update VISUAL_PROMPTS with a bounding box drawn around the target object
# in that reference image (format: [x1, y1, x2, y2] in pixel coordinates).
# ---------------------------------------------------------------------------
REFERENCE_IMAGE_PATH = "sample_image.png"

VISUAL_PROMPTS = dict(
    bboxes=np.array([[563, 431, 623, 582]]),
    cls=np.array([0]),
)

ID_TO_NAME = {0: "bottle"}


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

def pixel_to_world_on_plane(u: float, v: float, z_world: float):
    """Back-project pixel (u, v) onto the horizontal plane z = z_world.

    Returns a 3-element array [x, y, z] in robot base frame, or None if
    the ray does not intersect the plane.
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    camera = RealSenseLatestRGB(width=1280, height=720, fps=30)
    model  = YOLOE("yoloe-26x-seg.pt")

    window = "Live Bottle Detection — World Coordinates  (Q to quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    print("Starting live detection. Press Q to quit.")
    time.sleep(2.0)  # let the RealSense stream stabilize

    try:
        while True:
            img, _ = camera.get_latest()
            if img is None:
                time.sleep(0.01)
                continue

            result = model.predict(
                source=img,
                refer_image=REFERENCE_IMAGE_PATH,
                visual_prompts=VISUAL_PROMPTS,
                predictor=YOLOEVPSegPredictor,
                conf=0.25,
                verbose=False,
            )[0]

            display = img.copy()
            best    = None

            for box in (result.boxes or []):
                cls_id = int(box.cls.item())
                conf   = float(box.conf.item())
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                u = 0.5 * (x1 + x2)
                v = 0.5 * (y1 + y2)

                p_world = pixel_to_world_on_plane(u, v, BOTTLE_Z_WORLD)
                if p_world is not None:
                    p_world = p_world + SHIFT_VEC_BOTTLES

                det = {
                    "cls_id": cls_id, "conf": conf,
                    "bbox":   (x1, y1, x2, y2),
                    "uv":     (u, v),
                    "world":  p_world,
                }
                if best is None or conf > best["conf"]:
                    best = det

            if best is not None:
                x1, y1, x2, y2 = best["bbox"]
                u,  v           = best["uv"]
                p_world         = best["world"]
                obj_name        = ID_TO_NAME.get(best["cls_id"], f"cls_{best['cls_id']}")

                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(display, (int(u), int(v)), 4, (0, 255, 255), -1)

                if p_world is None:
                    label = f"{obj_name} {best['conf']:.2f} | world: n/a"
                else:
                    label = (
                        f"{obj_name} {best['conf']:.2f} | "
                        f"X:{p_world[0]:.3f}  Y:{p_world[1]:.3f}  Z:{p_world[2]:.3f}"
                    )

                text_y = y1 - 10 if y1 > 30 else y1 + 25
                cv2.putText(display, label, (x1, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(display, "No bottle detected", (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow(window, display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
