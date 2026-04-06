"""
app.py  —  GOD FILTER
Face filters with gesture control, face identity, and gesture training.

Run:  streamlit run app.py
"""

import os
import time
import cv2
import numpy as np
import mediapipe as mp
import streamlit as st
from datetime import datetime

from head_pose          import estimate_pose, draw_pose_hud
from blink_counter      import BlinkTracker, draw_blink_hud
from gesture_classifier import GestureCollector, GestureClassifier

GESTURE_LABELS = ["peace", "fist", "thumbs_up", "open_palm"]

st.set_page_config(page_title="GOD FILTER", layout="wide")

# ── Session state defaults ─────────────────────────────────────────────────────
def _init(key, val):
    if key not in st.session_state:
        st.session_state[key] = val

_init("current_filter",  0)
_init("recording",       False)
_init("snapshots",       [])
_init("frame_count",     0)
_init("snap_count",      0)
_init("do_snapshot",     False)
_init("smile_cd",        0)
_init("brow_cd",         0)
_init("palm_cd",         0)
_init("gesture_cd",      0)
_init("collecting",      False)
_init("collect_label",   GESTURE_LABELS[0])
_init("collect_counts",  {g: 0 for g in GESTURE_LABELS})
_init("writer",          None)
_init("writer_path",     None)

FILTER_NAMES  = ["Sunglasses", "Mustache", "Crown", "Horns"]
FILTER_EMOJIS = ["🕶️", "👨", "👑", "😈"]

os.makedirs("filters",   exist_ok=True)
os.makedirs("snapshots", exist_ok=True)
os.makedirs("recordings", exist_ok=True)
os.makedirs("models",    exist_ok=True)


# ── Resource loaders ──────────────────────────────────────────────────────────
@st.cache_resource
def load_face_mesh():
    return mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

@st.cache_resource
def load_hands():
    return mp.solutions.hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )

@st.cache_resource
def load_filters():
    paths = [
        "filters/sunglasses.png",
        "filters/mustache.png",
        "filters/crown.png",
        "filters/horns.png",
    ]
    imgs = []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img is None:
            img = np.zeros((10, 10, 4), dtype=np.uint8)
        imgs.append(img)
    return imgs

@st.cache_resource
def load_gesture_classifier():
    gc = GestureClassifier(smooth_window=7)
    gc.load()
    return gc

@st.cache_resource
def load_gesture_collector():
    return GestureCollector(min_samples_per_class=80)


face_mesh   = load_face_mesh()
hands_det   = load_hands()
filters     = load_filters()
g_clf       = load_gesture_classifier()
g_col       = load_gesture_collector()
blink_tracker = BlinkTracker()
prev_points = {}


# ── Helpers ───────────────────────────────────────────────────────────────────
def smooth_point(store, id_, x, y, alpha=0.3):
    if id_ not in store:
        store[id_] = (x, y)
    px, py = store[id_]
    sx = int(px * (1 - alpha) + x * alpha)
    sy = int(py * (1 - alpha) + y * alpha)
    store[id_] = (sx, sy)
    return sx, sy


def overlay_image(frame, img, x, y, w, h, angle=0):
    if img is None or w <= 0 or h <= 0:
        return frame
    img = cv2.resize(img, (w, h))
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    fh, fw = img.shape[:2]
    if angle != 0:
        M = cv2.getRotationMatrix2D((fw // 2, fh // 2), angle, 1)
        img = cv2.warpAffine(img, M, (fw, fh))
    if x < 0 or y < 0 or y + fh > frame.shape[0] or x + fw > frame.shape[1]:
        return frame
    b, g, r, a = cv2.split(img)
    mask = cv2.merge((a, a, a)) / 255.0
    roi  = frame[y:y + fh, x:x + fw].astype(np.float32)
    frame[y:y + fh, x:x + fw] = (
        roi * (1 - mask) + cv2.merge((b, g, r)).astype(np.float32) * mask
    ).astype(np.uint8)
    return frame


def detect_smile(lm, w, h, face_width):
    mw = abs(lm[291].x - lm[61].x) * w
    return (mw / (face_width + 1e-6)) > 0.44


def detect_eyebrow_raise(lm, h):
    lby = lm[70].y  * h
    rby = lm[300].y * h
    ley = lm[159].y * h
    rey = lm[386].y * h
    fh  = abs(lm[10].y - lm[152].y) * h
    return (ley - lby) > fh * 0.08 and (rey - rby) > fh * 0.08


def detect_open_palm(hand_lm):
    lm = hand_lm.landmark
    tips     = [4, 8, 12, 16, 20]
    knuckles = [2, 5,  9, 13, 17]
    extended = 0
    for tip, knk in zip(tips, knuckles):
        if tip == 4:
            extended += abs(lm[tip].x - lm[knk].x) > 0.06
        else:
            extended += lm[tip].y < lm[knk].y - 0.04
    return extended >= 4


def draw_rec_indicator(frame):
    cv2.circle(frame, (frame.shape[1] - 30, 30), 8, (0, 0, 255), -1)
    cv2.putText(frame, "REC", (frame.shape[1] - 65, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


# ── Frame processor ───────────────────────────────────────────────────────────
def process_frame(frame, current_filter, store, fidx):
    rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame.shape[:2]

    info = {
        "palm": False,
        "gesture": "Unknown",
        "gesture_conf": 0.0,
        "gesture_action": None,
        "yaw": 0.0, "pitch": 0.0, "roll": 0.0,
        "ear": 1.0, "bpm": 0, "blink": False,
    }

    # Hand detection (every 3rd frame to save CPU)
    if fidx % 3 == 0:
        hand_results = hands_det.process(rgb)
    else:
        hand_results = getattr(process_frame, "_last_hands", None)
    process_frame._last_hands = hand_results if fidx % 3 == 0 else getattr(process_frame, "_last_hands", None)

    if hand_results and hand_results.multi_hand_landmarks:
        for hand_lm in hand_results.multi_hand_landmarks:
            if detect_open_palm(hand_lm):
                info["palm"] = True

            if g_clf.is_trained:
                gesture, g_conf, g_action = g_clf.predict(hand_lm)
                info["gesture"]        = gesture
                info["gesture_conf"]   = g_conf
                info["gesture_action"] = g_action

            if st.session_state.collecting:
                label = st.session_state.collect_label
                cnt, _ = g_col.collect(hand_lm, label)
                st.session_state.collect_counts[label] = cnt

    # Face mesh
    face_results = face_mesh.process(rgb)
    if face_results.multi_face_landmarks:
        for face in face_results.multi_face_landmarks:
            lm = face.landmark

            x1, y1 = smooth_point(store, 1, int(lm[33].x * w),  int(lm[33].y * h))
            x2, y2 = smooth_point(store, 2, int(lm[263].x * w), int(lm[263].y * h))
            eye_dist   = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            face_width = int(eye_dist * 1.5)
            angle      = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            cx, cy     = (x1 + x2) // 2, (y1 + y2) // 2
            ox, oy     = cx - face_width // 2, cy - face_width // 4

            nx, ny = smooth_point(store, 3, int(lm[1].x * w),  int(lm[1].y * h))
            hx, hy = smooth_point(store, 4, int(lm[10].x * w), int(lm[10].y * h))

            # Head pose
            yaw, pitch, roll = estimate_pose(lm, w, h)
            info["yaw"], info["pitch"], info["roll"] = yaw, pitch, roll

            # Blink tracking
            blink_now, ear, bpm = blink_tracker.update(lm, w, h)
            info["blink"], info["ear"], info["bpm"] = blink_now, ear, bpm

            # Smile → sunglasses, brow raise → crown
            if detect_smile(lm, w, h, face_width):
                if st.session_state.smile_cd < fidx:
                    current_filter = 0
                    st.session_state.smile_cd = fidx + 30
            elif detect_eyebrow_raise(lm, h):
                if st.session_state.brow_cd < fidx:
                    current_filter = 2
                    st.session_state.brow_cd = fidx + 30

            # Draw selected filter
            if current_filter == 0:
                frame = overlay_image(frame, filters[0], ox, oy, face_width, int(face_width * 0.5), angle)
            elif current_filter == 1:
                frame = overlay_image(frame, filters[1], nx - face_width // 4, ny, face_width // 2, face_width // 4, 0)
            elif current_filter == 2:
                frame = overlay_image(frame, filters[2], hx - face_width // 2, hy - face_width // 2, face_width, face_width // 2, 0)
            elif current_filter == 3:
                frame = overlay_image(frame, filters[3], hx - face_width // 2, hy - face_width, face_width, face_width // 2, 0)

    return frame, current_filter, info


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("GOD FILTER")

    # Filter picker
    st.subheader("Filters")
    active = st.session_state.current_filter
    st.write(f"Active: {FILTER_EMOJIS[active]} {FILTER_NAMES[active]}")
    cols = st.columns(2)
    for i, (name, emoji) in enumerate(zip(FILTER_NAMES, FILTER_EMOJIS)):
        with cols[i % 2]:
            if st.button(f"{emoji} {name}", key=f"filter_{i}"):
                st.session_state.current_filter = i

    # Gesture trainer
    st.divider()
    st.subheader("Gesture Trainer")

    collect_label = st.selectbox(
        "Gesture", GESTURE_LABELS,
        index=GESTURE_LABELS.index(st.session_state.collect_label)
    )
    st.session_state.collect_label = collect_label

    for g in GESTURE_LABELS:
        c = st.session_state.collect_counts.get(g, 0)
        bar   = "█" * min(c // 8, 10)
        color = "🟢" if c >= 80 else "🟡" if c >= 40 else "🔴"
        st.markdown(f"{color} `{g:<12}` {bar} {c}/80")

    gc1, gc2 = st.columns(2)
    with gc1:
        btn_label = "⏹ Stop" if st.session_state.collecting else "⏺ Collect"
        if st.button(btn_label):
            st.session_state.collecting = not st.session_state.collecting
    with gc2:
        if st.button("🧠 Train", disabled=not g_col.is_ready()):
            with st.spinner("Training…"):
                acc = g_clf.train(g_col, verbose=True)
                g_clf.save()
            st.success(f"Done! Accuracy: {acc:.1%}")

    if g_clf.is_trained:
        st.success("Model ready")
    else:
        st.warning("No model — collect & train first")

    # Recording controls
    st.divider()
    st.subheader("Recording")

    if st.session_state.recording:
        st.error("🔴 Recording...")
    else:
        st.write("● Idle")

    r1, r2 = st.columns(2)
    with r1:
        rec_label = "⏹ Stop" if st.session_state.recording else "⏺ REC"
        if st.button(rec_label):
            st.session_state.recording = not st.session_state.recording
            if not st.session_state.recording:
                # Signal to release writer on next frame
                st.session_state["stop_recording"] = True
    with r2:
        if st.button("📸 Snap"):
            st.session_state.do_snapshot = True

    # Gallery
    st.divider()
    st.write(f"Snapshots: {st.session_state.snap_count}")
    if st.session_state.snapshots:
        for snap_path, snap_time in reversed(st.session_state.snapshots[-4:]):
            if os.path.exists(snap_path):
                st.image(snap_path, caption=snap_time, use_container_width=True)


# ── Main feed ─────────────────────────────────────────────────────────────────
col_title, col_stop = st.columns([4, 1])
with col_title:
    st.subheader(f"{FILTER_EMOJIS[st.session_state.current_filter]} Live Feed")
with col_stop:
    stop = st.button("■ Stop", type="secondary")

feed_placeholder = st.empty()

# ── Camera setup ──────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,   1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
cap.set(cv2.CAP_PROP_FPS,           30)
cap.set(cv2.CAP_PROP_BUFFERSIZE,    1)
cap.set(cv2.CAP_PROP_AUTOFOCUS,     1)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)

# Drain stale buffered frames and wait until we get a bright, stable frame.
# Cameras often return dark/black frames for the first ~10-15 reads while
# the sensor and auto-exposure settle — we skip all of them silently.
_warmup_start = time.time()
_stable        = False
while time.time() - _warmup_start < 3.0:          # max 3 s wait
    _ret, _frm = cap.read()
    if not _ret:
        continue
    # Consider the frame stable once the mean brightness is above a threshold
    if cv2.cvtColor(_frm, cv2.COLOR_BGR2GRAY).mean() > 20:
        _stable = True
        break

# Flush the buffer one more time so the first rendered frame is the freshest
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
for _ in range(2):
    cap.read()

writer  = None
fidx    = 0

# ── HUD drawing helpers ────────────────────────────────────────────────────────
def hud_panel(frame, lines, x, y, font_scale=0.52, thickness=1, pad=6):
    """
    Draw a semi-transparent dark panel with coloured text lines.
    lines = [(text, (B,G,R)), ...]
    Returns the bottom-y of the panel so the next panel can start there.
    """
    line_h  = int((font_scale * 28) + pad)
    panel_h = line_h * len(lines) + pad
    panel_w = max(
        cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0][0]
        for t, _ in lines
    ) + pad * 2

    # Clamp to frame
    x2 = min(x + panel_w, frame.shape[1] - 1)
    y2 = min(y + panel_h, frame.shape[0] - 1)

    overlay = frame[y:y2, x:x2].copy()
    cv2.rectangle(frame, (x, y), (x2, y2), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.35, frame[y:y2, x:x2], 0.65, 0, frame[y:y2, x:x2])

    for i, (text, color) in enumerate(lines):
        ty = y + pad + int(line_h * (i + 0.75))
        cv2.putText(frame, text, (x + pad, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

    return y2 + 6   # gap between panels


GESTURE_LABELS_EMOJI = {
    "peace": "V", "fist": "FIST", "thumbs_up": "THUMB",
    "open_palm": "PALM", "Unknown": "?"
}

# ── Main loop ─────────────────────────────────────────────────────────────────
prev_time = time.time()

while cap.isOpened() and not stop:
    ret, frame = cap.read()
    if not ret:
        st.error("Cannot access camera.")
        break

    frame = cv2.flip(frame, 1)
    fidx += 1
    st.session_state.frame_count += 1

    frame, active_filter, info = process_frame(
        frame, st.session_state.current_filter, prev_points, fidx
    )
    st.session_state.current_filter = active_filter

    h, w = frame.shape[:2]

    # ── Recording ─────────────────────────────────────────────────────────────
    if st.session_state.recording:
        if writer is None:
            ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = f"recordings/rec_{ts}.avi"
            writer   = cv2.VideoWriter(
                out_path,
                cv2.VideoWriter_fourcc(*"XVID"),
                30,
                (w, h),
            )
            st.session_state.writer_path = out_path
        writer.write(frame)
        # REC dot — top-right
        cv2.circle(frame, (w - 24, 20), 8, (0, 0, 220), -1)
        cv2.putText(frame, "REC", (w - 58, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 220), 1, cv2.LINE_AA)
    else:
        if writer is not None:
            writer.release()
            writer = None

    # ── Snapshot ──────────────────────────────────────────────────────────────
    palm_snap    = info["palm"]                         and st.session_state.palm_cd    < fidx
    gesture_snap = info["gesture_action"] == "snapshot" and st.session_state.gesture_cd < fidx

    if st.session_state.do_snapshot or palm_snap or gesture_snap:
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"snapshots/snap_{ts}.png"
        cv2.imwrite(path, frame)
        st.session_state.snapshots.append((path, ts))
        st.session_state.snap_count  += 1
        st.session_state.do_snapshot  = False
        st.session_state.palm_cd      = fidx + 90
        st.session_state.gesture_cd   = fidx + 90
        # Flash white border on snap
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (255, 255, 255), 6)

    # ── Gesture filter switching ───────────────────────────────────────────────
    g_action = info["gesture_action"]
    if g_action and st.session_state.gesture_cd < fidx:
        if g_action == "filter_next":
            st.session_state.current_filter = (st.session_state.current_filter + 1) % 4
            st.session_state.gesture_cd = fidx + 60
        elif g_action == "filter_prev":
            st.session_state.current_filter = (st.session_state.current_filter - 1) % 4
            st.session_state.gesture_cd = fidx + 60
        elif g_action == "toggle_rec":
            st.session_state.recording = not st.session_state.recording
            st.session_state.gesture_cd = fidx + 90

    # ── FPS ───────────────────────────────────────────────────────────────────
    now      = time.time()
    fps      = 1.0 / max(now - prev_time, 1e-6)
    prev_time = now

    # ── HUD — left column, stacked panels ─────────────────────────────────────
    next_y = 10

    # Panel 1 — Head pose
    next_y = hud_panel(frame, [
        (f"YAW   {info['yaw']:+.0f}",   (100, 210, 255)),
        (f"PITCH {info['pitch']:+.0f}", (100, 255, 200)),
        (f"ROLL  {info['roll']:+.0f}",  (255, 200, 100)),
    ], x=10, y=next_y)

    # Panel 2 — Blink / EAR
    fatigue_label, fatigue_color = blink_tracker.fatigue_level(info["bpm"])
    next_y = hud_panel(frame, [
        (f"EAR    {info['ear']:.3f}",        (180, 180, 180)),
        (f"BLINKS {blink_tracker.total_blinks}", (180, 180, 180)),
        (f"BPM    {info['bpm']}  {fatigue_label}", fatigue_color),
    ], x=10, y=next_y)

    # Panel 3 — Gesture (only when model trained)
    if g_clf.is_trained:
        gesture      = info["gesture"]
        gesture_conf = info["gesture_conf"]
        g_color      = (80, 255, 120) if gesture != "Unknown" else (130, 130, 130)
        tag          = GESTURE_LABELS_EMOJI.get(gesture, "?")
        gesture_lines = [
            (f"[{tag}] {gesture}", g_color),
            (f"CONF   {gesture_conf:.0%}", (180, 180, 180)),
        ]
        if info["gesture_action"]:
            gesture_lines.append((f"-> {info['gesture_action']}", (255, 200, 60)))
        next_y = hud_panel(frame, gesture_lines, x=10, y=next_y)

    # Panel 4 — FPS  (small, bottom of left column)
    hud_panel(frame, [(f"FPS {fps:.0f}", (160, 160, 160))],
              x=10, y=next_y, font_scale=0.42)

    # ── Display — full resolution, no downscale ────────────────────────────────
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    feed_placeholder.image(rgb, channels="RGB", use_container_width=True)

cap.release()
if writer:
    writer.release()