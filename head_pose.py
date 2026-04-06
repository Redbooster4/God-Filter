"""
head_pose.py
─────────────────────────────────────────────────────────
Estimates Yaw (left-right), Pitch (up-down), Roll (tilt)
from MediaPipe FaceMesh landmarks using PnP solve.

Returns angles in degrees.
  Yaw   > 0  → face turned right
  Pitch > 0  → face tilted up
  Roll  > 0  → face tilted clockwise
"""

import cv2
import numpy as np


# 3-D reference points of a generic face model (OpenCV coords)
_MODEL_POINTS = np.array([
    (0.0,    0.0,    0.0),     # Nose tip           – lm 1
    (0.0,   -330.0, -65.0),    # Chin               – lm 152
    (-225.0,  170.0, -135.0),  # Left eye corner    – lm 33
    (225.0,   170.0, -135.0),  # Right eye corner   – lm 263
    (-150.0, -150.0, -125.0),  # Left mouth corner  – lm 61
    (150.0,  -150.0, -125.0),  # Right mouth corner – lm 291
], dtype=np.float64)

_LM_IDS = [1, 152, 33, 263, 61, 291]


def _camera_matrix(w, h):
    focal = w
    cx, cy = w / 2, h / 2
    return np.array([
        [focal, 0,  cx],
        [0, focal,  cy],
        [0,     0,   1],
    ], dtype=np.float64)


def estimate_pose(landmarks, frame_w, frame_h):
    """
    Parameters
    ----------
    landmarks : list of MediaPipe NormalizedLandmark
    frame_w, frame_h : int

    Returns
    -------
    yaw, pitch, roll : float  (degrees)
    """
    img_pts = np.array([
        (landmarks[i].x * frame_w, landmarks[i].y * frame_h)
        for i in _LM_IDS
    ], dtype=np.float64)

    cam_mat  = _camera_matrix(frame_w, frame_h)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rvec, tvec = cv2.solvePnP(
        _MODEL_POINTS, img_pts, cam_mat, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return 0.0, 0.0, 0.0

    rmat, _ = cv2.Rodrigues(rvec)
    # Decompose rotation matrix → Euler angles
    sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        roll  = np.degrees(np.arctan2( rmat[2, 1], rmat[2, 2]))
        pitch = np.degrees(np.arctan2(-rmat[2, 0], sy))
        yaw   = np.degrees(np.arctan2( rmat[1, 0], rmat[0, 0]))
    else:
        roll  = np.degrees(np.arctan2(-rmat[1, 2], rmat[1, 1]))
        pitch = np.degrees(np.arctan2(-rmat[2, 0], sy))
        yaw   = 0.0

    return round(yaw, 1), round(pitch, 1), round(roll, 1)


def draw_pose_hud(frame, yaw, pitch, roll, x=16, y=150):
    """Draws pose angles as a mini HUD block onto frame (in-place)."""
    labels = [
        (f"YAW   {yaw:+.0f}deg",   (100, 200, 255)),
        (f"PITCH {pitch:+.0f}deg", (100, 255, 200)),
        (f"ROLL  {roll:+.0f}deg",  (255, 200, 100)),
    ]
    for i, (text, color) in enumerate(labels):
        cv2.putText(frame, text,
                    (x, y + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)
