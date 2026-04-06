"""
gesture_classifier.py
─────────────────────────────────────────────────────────
Custom ML gesture classifier trained on MediaPipe hand
landmark coordinates (21 points × 2 = 42 features).

Gestures
────────
  0 → thumbs_up
  1 → peace
  2 → fist
  3 → ok_sign
  4 → open_palm
  5 → point

Pipeline
────────
  1. COLLECT  → GestureCollector.collect(landmarks, label)
  2. TRAIN    → GestureClassifier.train(collector.data)
  3. PREDICT  → GestureClassifier.predict(landmarks) → (label, confidence)
  4. SAVE/LOAD→ GestureClassifier.save/load("gesture_model.pkl")

The feature vector is normalized relative to the wrist (lm 0)
so position/scale invariance is built in.
"""

import os
import pickle
import numpy as np
import cv2
from collections import defaultdict, deque

# ── Sklearn pipeline ─────────────────────────────────────
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing   import StandardScaler, LabelEncoder
from sklearn.pipeline        import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics         import classification_report

MODEL_PATH = "gesture_model.pkl"

GESTURE_LABELS = [
    "thumbs_up",
    "peace",
    "fist",
    "ok_sign",
    "open_palm",
    "point",
]

# Action each gesture triggers in the app
GESTURE_ACTIONS = {
    "thumbs_up":  "snapshot",
    "peace":      "filter_next",
    "fist":       "filter_prev",
    "ok_sign":    "toggle_rec",
    "open_palm":  "snapshot",     # same as button
    "point":      None,           # no action, just display
}

# Emoji for HUD display
GESTURE_EMOJI = {
    "thumbs_up": "👍",
    "peace":     "✌️",
    "fist":      "✊",
    "ok_sign":   "👌",
    "open_palm": "🤚",
    "point":     "☝️",
    "Unknown":   "❓",
}


# ── Feature extraction ────────────────────────────────────
def landmarks_to_features(hand_landmarks):
    """
    Convert 21 MediaPipe hand landmarks → 42-d normalized feature vector.

    Normalization:
      - Translate so wrist (lm 0) is at origin
      - Scale so the distance from wrist to middle-finger MCP (lm 9) = 1
    """
    lm = hand_landmarks.landmark
    # Raw coords
    pts = np.array([[l.x, l.y] for l in lm], dtype=np.float32)  # (21, 2)

    # Translate to wrist
    pts -= pts[0]

    # Scale by wrist-to-middle-MCP distance
    scale = np.linalg.norm(pts[9]) + 1e-8
    pts  /= scale

    return pts.flatten()   # 42-d vector


# ── Data collector ────────────────────────────────────────
class GestureCollector:
    """
    Accumulates labeled samples for training.

    Usage:
        collector = GestureCollector()
        # Per frame when user holds a gesture:
        collector.collect(hand_landmarks, "thumbs_up")
        # After enough samples:
        classifier.train(collector)
    """

    def __init__(self, min_samples_per_class=60):
        self.X   = []
        self.y   = []
        self.min_samples = min_samples_per_class
        self._counts = defaultdict(int)

    def collect(self, hand_landmarks, label):
        """Add one sample. Returns (count_for_label, is_ready)."""
        feat = landmarks_to_features(hand_landmarks)
        self.X.append(feat)
        self.y.append(label)
        self._counts[label] += 1
        ready = all(
            self._counts.get(g, 0) >= self.min_samples
            for g in GESTURE_LABELS
        )
        return self._counts[label], ready

    def counts(self):
        return dict(self._counts)

    def is_ready(self):
        return all(
            self._counts.get(g, 0) >= self.min_samples
            for g in GESTURE_LABELS
        )

    def clear(self):
        self.X.clear()
        self.y.clear()
        self._counts.clear()


# ── Classifier ────────────────────────────────────────────
class GestureClassifier:
    """
    RandomForest-based gesture classifier with a StandardScaler pre-processor.

    Smoothing: majority vote over last N predictions prevents flickering.
    """

    CONFIDENCE_THRESHOLD = 0.55

    def __init__(self, smooth_window=7):
        self._pipeline   = None
        self._encoder    = LabelEncoder()
        self._trained    = False
        self._history    = deque(maxlen=smooth_window)

    # ── Training ──────────────────────────────────────────
    def train(self, collector: GestureCollector, verbose=True):
        """
        Train on data collected by GestureCollector.
        Returns cross-validation accuracy.
        """
        X = np.array(collector.X)
        y = np.array(collector.y)

        self._encoder.fit(GESTURE_LABELS)
        y_enc = self._encoder.transform(y)

        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            )),
        ])

        # Cross-val score
        scores = cross_val_score(self._pipeline, X, y_enc, cv=5, scoring="accuracy")
        acc    = scores.mean()

        # Final fit on all data
        self._pipeline.fit(X, y_enc)
        self._trained = True

        if verbose:
            y_pred = self._pipeline.predict(X)
            print("\n── Gesture Classifier Training Report ──")
            print(classification_report(
                y_enc, y_pred,
                target_names=self._encoder.classes_,
            ))
            print(f"Cross-val accuracy: {acc:.2%}  (±{scores.std():.2%})")

        return acc

    # ── Inference ─────────────────────────────────────────
    def predict(self, hand_landmarks):
        """
        Returns (gesture_label: str, confidence: float, action: str|None).
        Returns ("Unknown", 0, None) if not trained or low confidence.
        """
        if not self._trained:
            return "Unknown", 0.0, None

        feat  = landmarks_to_features(hand_landmarks).reshape(1, -1)
        proba = self._pipeline.predict_proba(feat)[0]
        idx   = int(np.argmax(proba))
        conf  = float(proba[idx])

        if conf < self.CONFIDENCE_THRESHOLD:
            label = "Unknown"
        else:
            label = self._encoder.inverse_transform([idx])[0]

        # Smoothing: majority vote in recent window
        self._history.append(label)
        smoothed = max(set(self._history), key=list(self._history).count)

        action = GESTURE_ACTIONS.get(smoothed)
        return smoothed, round(conf, 3), action

    # ── Persistence ──────────────────────────────────────
    def save(self, path=MODEL_PATH):
        with open(path, "wb") as f:
            pickle.dump({
                "pipeline": self._pipeline,
                "encoder":  self._encoder,
                "trained":  self._trained,
            }, f)
        print(f"[GestureClassifier] Saved → {path}")

    def load(self, path=MODEL_PATH):
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._pipeline = data["pipeline"]
        self._encoder  = data["encoder"]
        self._trained  = data["trained"]
        print(f"[GestureClassifier] Loaded ← {path}")
        return True

    @property
    def is_trained(self):
        return self._trained


# ── HUD drawing ──────────────────────────────────────────
def draw_gesture_hud(frame, gesture, confidence, action, x=16, y=420):
    emoji  = GESTURE_EMOJI.get(gesture, "❓")
    color  = (80, 255, 120) if gesture != "Unknown" else (100, 100, 100)
    lines  = [
        (f"GESTURE {gesture}",          color),
        (f"        {confidence:.0%}",   (180, 180, 180)),
    ]
    if action:
        lines.append((f"ACTION  {action.upper()}", (255, 200, 80)))

    for i, (text, col) in enumerate(lines):
        cv2.putText(frame, text,
                    (x, y + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1)
                        # Draw emoji bigger above text
        cv2.putText(frame, emoji,
                    (x, y - 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    color,
                    2,
                    cv2.LINE_AA)

        # Optional background panel (clean UI look)
        cv2.rectangle(frame,
                    (x-10, y-40),
                    (x+200, y + len(lines)*22 + 5),
                    (30,30,30),
                    1)
        return frame