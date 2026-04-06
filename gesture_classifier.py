"""
gesture_classifier.py
─────────────────────
Custom ML gesture classifier trained on MediaPipe hand
landmark coordinates (21 points × 2 = 42 features).

Gestures:  peace, fist, thumbs_up, open_palm

Pipeline:
  1. COLLECT  → GestureCollector.collect(landmarks, label)
  2. TRAIN    → GestureClassifier.train(collector)
  3. PREDICT  → GestureClassifier.predict(landmarks) → (label, confidence, action)
  4. SAVE/LOAD→ GestureClassifier.save() / .load()
"""

import os
import pickle
import numpy as np
from collections import defaultdict, deque

from sklearn.ensemble        import RandomForestClassifier
from sklearn.preprocessing   import StandardScaler, LabelEncoder
from sklearn.pipeline        import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics         import classification_report

MODEL_PATH = "gesture_model.pkl"

GESTURE_LABELS = ["peace", "fist", "thumbs_up", "open_palm"]

GESTURE_ACTIONS = {
    "thumbs_up":  "snapshot",
    "peace":      "filter_next",
    "fist":       "filter_prev",
    "open_palm":  "snapshot",
}


# ── Feature extraction ────────────────────────────────────────────────────────
def landmarks_to_features(hand_landmarks):
    """
    21 MediaPipe landmarks → 42-d normalized vector.
    Translates to wrist origin, scales by wrist-to-middle-MCP distance.
    """
    lm  = hand_landmarks.landmark
    pts = np.array([[l.x, l.y] for l in lm], dtype=np.float32)
    pts -= pts[0]
    pts /= (np.linalg.norm(pts[9]) + 1e-8)
    return pts.flatten()


# ── Data collector ────────────────────────────────────────────────────────────
class GestureCollector:
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
        ready = self.is_ready()
        return self._counts[label], ready

    def counts(self):
        return dict(self._counts)

    def is_ready(self):
        return all(self._counts.get(g, 0) >= self.min_samples for g in GESTURE_LABELS)

    def clear(self):
        self.X.clear()
        self.y.clear()
        self._counts.clear()


# ── Classifier ────────────────────────────────────────────────────────────────
class GestureClassifier:
    CONFIDENCE_THRESHOLD = 0.55

    def __init__(self, smooth_window=7):
        self._pipeline = None
        self._encoder  = LabelEncoder()
        self._trained  = False
        self._history  = deque(maxlen=smooth_window)

    def train(self, collector: GestureCollector, verbose=True):
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

        scores = cross_val_score(self._pipeline, X, y_enc, cv=5, scoring="accuracy")
        acc    = scores.mean()
        self._pipeline.fit(X, y_enc)
        self._trained = True

        if verbose:
            y_pred = self._pipeline.predict(X)
            print("\n── Gesture Classifier Training Report ──")
            print(classification_report(y_enc, y_pred, target_names=self._encoder.classes_))
            print(f"Cross-val accuracy: {acc:.2%}  (±{scores.std():.2%})")

        return acc

    def predict(self, hand_landmarks):
        if not self._trained:
            return "Unknown", 0.0, None

        feat  = landmarks_to_features(hand_landmarks).reshape(1, -1)
        proba = self._pipeline.predict_proba(feat)[0]
        idx   = int(np.argmax(proba))
        conf  = float(proba[idx])

        label = self._encoder.inverse_transform([idx])[0] if conf >= self.CONFIDENCE_THRESHOLD else "Unknown"
        self._history.append(label)
        smoothed = max(set(self._history), key=list(self._history).count)

        return smoothed, round(conf, 3), GESTURE_ACTIONS.get(smoothed)

    def save(self, path=MODEL_PATH):
        with open(path, "wb") as f:
            pickle.dump({"pipeline": self._pipeline, "encoder": self._encoder, "trained": self._trained}, f)

    def load(self, path=MODEL_PATH):
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._pipeline = data["pipeline"]
        self._encoder  = data["encoder"]
        self._trained  = data["trained"]
        return True

    @property
    def is_trained(self):
        return self._trained