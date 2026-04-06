"""
blink_counter.py
─────────────────────────────────────────────────────────
Tracks blink rate using Eye Aspect Ratio (EAR).

EAR = (‖p2−p6‖ + ‖p3−p5‖) / (2 · ‖p1−p4‖)
     where p1..p6 are the 6 eye landmarks (horizontal + vertical pairs)

EAR < threshold for N consecutive frames → blink detected.

Also computes blinks-per-minute from a rolling 60-second window.
"""

import time
from collections import deque
import numpy as np


# MediaPipe FaceMesh landmark IDs for each eye
# Format: (outer, inner, top1, top2, bottom1, bottom2)
LEFT_EYE  = [33, 133, 160, 158, 153, 144]
RIGHT_EYE = [362, 263, 387, 385, 380, 373]

EAR_THRESHOLD   = 0.22   # below this = eye closing
CONSEC_FRAMES   = 2      # must be closed for this many frames to count as blink


def _ear(landmarks, ids, w, h):
    """Compute Eye Aspect Ratio for one eye."""
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in ids]

    # Vertical distances
    v1 = np.linalg.norm(np.array(pts[2]) - np.array(pts[5]))
    v2 = np.linalg.norm(np.array(pts[3]) - np.array(pts[4]))
    # Horizontal distance
    hz = np.linalg.norm(np.array(pts[0]) - np.array(pts[1]))

    return (v1 + v2) / (2.0 * hz + 1e-6)


class BlinkTracker:
    def __init__(self):
        self.total_blinks   = 0
        self._consec        = 0          # consecutive frames below threshold
        self._blink_times   = deque()    # timestamps of each blink (for BPM)
        self._last_ear      = 1.0

    def update(self, landmarks, w, h):
        """
        Call once per frame.
        Returns (blink_happened_this_frame: bool, current_ear: float, bpm: float)
        """
        left  = _ear(landmarks, LEFT_EYE,  w, h)
        right = _ear(landmarks, RIGHT_EYE, w, h)
        avg_ear = (left + right) / 2.0
        self._last_ear = avg_ear

        blink_now = False

        if avg_ear < EAR_THRESHOLD:
            self._consec += 1
        else:
            if self._consec >= CONSEC_FRAMES:
                self.total_blinks += 1
                self._blink_times.append(time.time())
                blink_now = True
            self._consec = 0

        # Prune timestamps older than 60 s
        now = time.time()
        while self._blink_times and now - self._blink_times[0] > 60:
            self._blink_times.popleft()

        bpm = len(self._blink_times)   # blinks in last 60 s = blinks per minute
        return blink_now, round(avg_ear, 3), bpm

    @property
    def ear(self):
        return self._last_ear

    def fatigue_level(self, bpm):
        """
        Returns a string label based on blink rate.
        Normal: 12-20 bpm.  Low → focused or fatigued.  High → irritated.
        """
        if bpm < 8:
            return "FATIGUED", (0, 80, 255)
        elif bpm > 25:
            return "IRRITATED", (0, 180, 255)
        else:
            return "NORMAL", (80, 255, 120)


def draw_blink_hud(frame, tracker, bpm, x=16, y=240):
    label, color = tracker.fatigue_level(bpm)
    lines = [
        (f"EAR   {tracker.ear:.3f}",      (180, 180, 180)),
        (f"BLINKS {tracker.total_blinks}", (180, 180, 180)),
        (f"BPM   {bpm}  [{label}]",        color),
    ]
    for i, (text, col) in enumerate(lines):
        cv2.putText(frame, text,
                    (x, y + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1)


# cv2 import needed for draw function
import cv2