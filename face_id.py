"""
face_id.py
─────────────────────────────────────────────────────────
Face recognition using OpenCV DNN + custom embeddings.

Uses OpenCV's face detector + a lightweight face embedder
(openface / nn4.small2) to produce 128-d face embeddings,
then compares via cosine similarity — no face_recognition
library needed, works cross-platform.

If the official models aren't present it falls back to a
landmark-geometry embedding (always works, lower accuracy).

Workflow
────────
1. Call enroll(frame, name) to register a person.
2. Call identify(frame) → returns (name, confidence) or ("Unknown", 0).
3. Call is_authorized(frame, threshold) → bool.

Model files expected at:
  models/deploy.prototxt
  models/res10_300x300_ssd_iter_140000.caffemodel
  models/openface.nn4.small2.v1.t7

Download links printed on first run if missing.
"""

import os
import cv2
import numpy as np
from pathlib import Path

MODEL_DIR    = Path("models")
PROTO_PATH   = MODEL_DIR / "deploy.prototxt"
CAFFE_PATH   = MODEL_DIR / "res10_300x300_ssd_iter_140000.caffemodel"
OPENFACE_PATH= MODEL_DIR / "openface.nn4.small2.v1.t7"

DOWNLOAD_HINTS = """
─── Missing model files ─────────────────────────────────
Please download:
1. deploy.prototxt
   https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt

2. res10_300x300_ssd_iter_140000.caffemodel
   https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel

3. openface.nn4.small2.v1.t7
   https://storage.cmusatyalab.org/openface-models/nn4.small2.v1.t7

Place all 3 in a  models/  folder next to your script.
─────────────────────────────────────────────────────────
"""


def _cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


class FaceIDSystem:
    def __init__(self):
        self._detector  = None
        self._embedder  = None
        self._db        = {}        # name → list of embedding vectors
        self._ready     = False
        self._mode      = "none"    # "dnn" | "geometry" | "none"
        self._load_models()

    # ── Model loading ─────────────────────────────────────
    def _load_models(self):
        MODEL_DIR.mkdir(exist_ok=True)

        if PROTO_PATH.exists() and CAFFE_PATH.exists():
            try:
                self._detector = cv2.dnn.readNetFromCaffe(
                    str(PROTO_PATH), str(CAFFE_PATH)
                )
                if OPENFACE_PATH.exists():
                    self._embedder = cv2.dnn.readNetFromTorch(str(OPENFACE_PATH))
                    self._mode  = "dnn"
                    self._ready = True
                    print("[FaceID] DNN mode — detector + embedder loaded.")
                else:
                    self._mode  = "geometry"
                    self._ready = True
                    print("[FaceID] Geometry-fallback mode (no openface model).")
            except Exception as e:
                print(f"[FaceID] Model load error: {e}")
        else:
            print(DOWNLOAD_HINTS)
            self._mode  = "geometry"
            self._ready = True
            print("[FaceID] Geometry-fallback mode (model files missing).")

    # ── Face detection ────────────────────────────────────
    def _detect_face(self, frame):
        """Returns cropped face ROI or None."""
        h, w = frame.shape[:2]

        if self._detector is not None:
            blob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)), 1.0,
                (300, 300), (104, 117, 123)
            )
            self._detector.setInput(blob)
            dets = self._detector.forward()
            for i in range(dets.shape[2]):
                conf = dets[0, 0, i, 2]
                if conf > 0.6:
                    box = dets[0, 0, i, 3:7] * np.array([w, h, w, h])
                    x1, y1, x2, y2 = box.astype(int)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    if x2 > x1 and y2 > y1:
                        return frame[y1:y2, x1:x2]
        else:
            # Fallback: assume center 60% of frame is the face
            m = 0.2
            return frame[int(h*m):int(h*(1-m)), int(w*m):int(w*(1-m))]
        return None

    # ── Embedding ─────────────────────────────────────────
    def _embed(self, face_roi, mp_landmarks=None):
        """Returns a 1-D embedding vector."""
        if self._mode == "dnn" and self._embedder is not None:
            face_resized = cv2.resize(face_roi, (96, 96))
            blob = cv2.dnn.blobFromImage(
                face_resized, 1.0 / 255,
                (96, 96), (0, 0, 0), swapRB=True, crop=False
            )
            self._embedder.setInput(blob)
            return self._embedder.forward().flatten()

        # Geometry fallback: use MediaPipe landmark ratios as pseudo-embedding
        if mp_landmarks is not None:
            pts = [(lm.x, lm.y) for lm in mp_landmarks]
            vec = np.array(pts).flatten()
            # Normalize relative to eye midpoint
            mid_x = (mp_landmarks[33].x + mp_landmarks[263].x) / 2
            mid_y = (mp_landmarks[33].y + mp_landmarks[263].y) / 2
            vec[0::2] -= mid_x
            vec[1::2] -= mid_y
            return vec

        # Last resort: raw pixel histogram
        gray  = cv2.cvtColor(cv2.resize(face_roi, (64, 64)), cv2.COLOR_BGR2GRAY)
        hist, _ = np.histogram(gray.flatten(), bins=128, range=(0, 256))
        return hist.astype(np.float32) / (hist.sum() + 1e-8)

    # ── Public API ────────────────────────────────────────
    def enroll(self, frame, name, mp_landmarks=None):
        """Register a face from frame under `name`. Call 3-5 times for best results."""
        face = self._detect_face(frame)
        if face is None:
            return False, "No face detected"
        emb = self._embed(face, mp_landmarks)
        self._db.setdefault(name, []).append(emb)
        return True, f"Enrolled '{name}' ({len(self._db[name])} samples)"

    def identify(self, frame, mp_landmarks=None):
        """
        Returns (name: str, confidence: float 0-1).
        name = "Unknown" if no match above threshold.
        """
        if not self._db:
            return "Unknown", 0.0

        face = self._detect_face(frame)
        if face is None:
            return "Unknown", 0.0

        emb      = self._embed(face, mp_landmarks)
        best_name, best_score = "Unknown", 0.0

        for name, samples in self._db.items():
            scores = [_cosine_sim(emb, s) for s in samples]
            avg    = float(np.mean(scores))
            if avg > best_score:
                best_score, best_name = avg, name

        if best_score < 0.55:
            return "Unknown", round(best_score, 3)
        return best_name, round(best_score, 3)

    def is_authorized(self, frame, authorized_names, threshold=0.55, mp_landmarks=None):
        name, conf = self.identify(frame, mp_landmarks)
        return name in authorized_names and conf >= threshold, name, conf

    def enrolled_names(self):
        return list(self._db.keys())

    def clear(self):
        self._db.clear()


def draw_id_hud(frame, name, confidence, authorized, x=16, y=330):
    color = (80, 255, 120) if authorized else (50, 50, 255)
    status = "AUTHORIZED" if authorized else "UNAUTHORIZED"
    lines = [
        (f"ID    {name}",              color),
        (f"CONF  {confidence:.0%}",    (180, 180, 180)),
        (f"      {status}",            color),
    ]
    for i, (text, col) in enumerate(lines):
        cv2.putText(frame, text,
                    (x, y + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1)
