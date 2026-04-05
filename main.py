import cv2
import mediapipe as mp
import numpy as np
import time
import streamlit as st
from datetime import datetime
import os

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GOD FILTER",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #0a0a0f;
    color: #e8e4dc;
}

.stApp {
    background: #0a0a0f;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f0f18 !important;
    border-right: 1px solid #1e1e2e;
}

[data-testid="stSidebar"] * {
    color: #e8e4dc !important;
}

/* Title */
.god-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.4rem;
    letter-spacing: -1px;
    background: linear-gradient(135deg, #ff6b35, #f7c59f, #ffffd1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0;
    line-height: 1;
}

.god-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #555577;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 4px;
    margin-bottom: 24px;
}

/* Filter buttons */
.stButton > button {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    background: #13131f !important;
    color: #e8e4dc !important;
    border: 1px solid #2a2a3e !important;
    border-radius: 4px !important;
    padding: 10px 16px !important;
    width: 100% !important;
    transition: all 0.2s !important;
}

.stButton > button:hover {
    background: #1e1e30 !important;
    border-color: #ff6b35 !important;
    color: #ff6b35 !important;
}

/* Active filter pill */
.active-pill {
    display: inline-block;
    background: #ff6b35;
    color: #0a0a0f;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 2px;
    margin-bottom: 16px;
}

/* Recording badge */
.rec-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: #1a0808;
    border: 1px solid #ff2222;
    color: #ff4444;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 6px 14px;
    border-radius: 2px;
}

.rec-dot {
    width: 8px;
    height: 8px;
    background: #ff2222;
    border-radius: 50%;
    animation: blink 1s infinite;
}

@keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.2; }
}

.idle-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: #0f1a0f;
    border: 1px solid #226622;
    color: #44aa44;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 6px 14px;
    border-radius: 2px;
}

/* Snapshot gallery */
.gallery-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: #555577;
    margin-bottom: 12px;
    margin-top: 24px;
    border-top: 1px solid #1e1e2e;
    padding-top: 16px;
}

/* Stats */
.stat-box {
    background: #13131f;
    border: 1px solid #1e1e2e;
    border-radius: 4px;
    padding: 12px 16px;
    text-align: center;
}

.stat-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.4rem;
    font-weight: 600;
    color: #ff6b35;
}

.stat-label {
    font-size: 0.65rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #555577;
    margin-top: 2px;
}

/* Feed frame */
.feed-container {
    border: 1px solid #1e1e2e;
    border-radius: 4px;
    overflow: hidden;
}

hr {
    border-color: #1e1e2e !important;
}

/* Selectbox / radio */
[data-testid="stSelectbox"] label,
[data-testid="stRadio"] label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: #555577 !important;
}
</style>
""", unsafe_allow_html=True)

# ─── SESSION STATE ─────────────────────────────────────────────────────────────
if "current_filter" not in st.session_state:
    st.session_state.current_filter = 0
if "recording" not in st.session_state:
    st.session_state.recording = False
if "snapshots" not in st.session_state:
    st.session_state.snapshots = []
if "frame_count" not in st.session_state:
    st.session_state.frame_count = 0
if "snap_count" not in st.session_state:
    st.session_state.snap_count = 0

FILTER_NAMES = ["Sunglasses", "Mustache", "Crown", "Dog Ears"]
FILTER_EMOJIS = ["🕶️", "👨", "👑", "🐶"]

os.makedirs("filters", exist_ok=True)
os.makedirs("snapshots", exist_ok=True)

# ─── MEDIAPIPE ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_face_mesh():
    mp_face_mesh = mp.solutions.face_mesh
    return mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

@st.cache_resource
def load_filters():
    paths = [
        "filters/sunglasses.png",
        "filters/mustache.png",
        "filters/crown.png",
        "filters/dog_ears.png"
    ]
    imgs = []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        imgs.append(img)
    return imgs

face_mesh = load_face_mesh()
filters = load_filters()
prev_points = {}

# ─── HELPERS ───────────────────────────────────────────────────────────────────
def smooth_point(store, id, x, y):
    if id not in store:
        store[id] = (x, y)
    px, py = store[id]
    sx = int(px * 0.7 + x * 0.3)
    sy = int(py * 0.7 + y * 0.3)
    store[id] = (sx, sy)
    return sx, sy

def overlay(frame, img, x, y, w, h, angle):
    if img is None or w <= 0 or h <= 0:
        return frame
    img = cv2.resize(img, (w, h))
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    fh, fw, _ = img.shape
    center = (fw // 2, fh // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1)
    img = cv2.warpAffine(img, M, (fw, fh))
    if x < 0 or y < 0:
        return frame
    if y + fh > frame.shape[0] or x + fw > frame.shape[1]:
        return frame
    b, g, r, a = cv2.split(img)
    mask = cv2.merge((a, a, a)) / 255.0
    rgb = cv2.merge((b, g, r))
    roi = frame[y:y + fh, x:x + fw].astype(np.float32)
    blended = roi * (1 - mask) + rgb.astype(np.float32) * mask
    frame[y:y + fh, x:x + fw] = blended.astype(np.uint8)
    return frame

def process_frame(frame, current_filter, store):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    mouth_open_val = 0.0

    if results.multi_face_landmarks:
        for face in results.multi_face_landmarks:
            lm = face.landmark
            h, w, _ = frame.shape

            left, right = lm[33], lm[263]
            x1, y1 = int(left.x * w), int(left.y * h)
            x2, y2 = int(right.x * w), int(right.y * h)
            x1, y1 = smooth_point(store, 1, x1, y1)
            x2, y2 = smooth_point(store, 2, x2, y2)

            width = int(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * 1.5)
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            x = int((x1 + x2) / 2 - width / 2)
            y = int((y1 + y2) / 2 - width / 4)

            nose = lm[1]
            nx, ny = int(nose.x * w), int(nose.y * h)
            nx, ny = smooth_point(store, 3, nx, ny)

            head = lm[10]
            hx, hy = int(head.x * w), int(head.y * h)
            hx, hy = smooth_point(store, 4, hx, hy)

            top_lip, bottom_lip = lm[13], lm[14]
            mouth_open_val = abs(top_lip.y - bottom_lip.y)

            if mouth_open_val > 0.02:
                current_filter = 3

            if current_filter == 0:
                frame = overlay(frame, filters[0], x, y, width, int(width * 0.5), angle)
            elif current_filter == 1:
                frame = overlay(frame, filters[1], nx - width // 4, ny, width // 2, width // 4, 0)
            elif current_filter == 2:
                frame = overlay(frame, filters[2], hx - width // 2, hy - width // 2, width, width // 2, 0)
            elif current_filter == 3:
                frame = overlay(frame, filters[3], hx - width // 2, hy - width, width, width // 2, 0)

    return frame, current_filter, mouth_open_val

# ─── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="god-title">GOD<br>FILTER</div>', unsafe_allow_html=True)
    st.markdown('<div class="god-sub">Face AR · v1.0</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="gallery-title" style="border-top:none;padding-top:0;margin-top:0">Active Filter</div>', unsafe_allow_html=True)

    active_name = FILTER_NAMES[st.session_state.current_filter]
    active_emoji = FILTER_EMOJIS[st.session_state.current_filter]
    st.markdown(f'<div class="active-pill">{active_emoji} {active_name}</div>', unsafe_allow_html=True)

    cols = st.columns(2)
    for i, (name, emoji) in enumerate(zip(FILTER_NAMES, FILTER_EMOJIS)):
        col = cols[i % 2]
        with col:
            if st.button(f"{emoji} {name}", key=f"filter_{i}"):
                st.session_state.current_filter = i

    st.markdown("---")
    st.markdown('<div class="gallery-title" style="border-top:none;padding-top:0;margin-top:0">Recording</div>', unsafe_allow_html=True)

    if st.session_state.recording:
        st.markdown('<div class="rec-badge"><div class="rec-dot"></div>REC ACTIVE</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="idle-badge">● IDLE</div>', unsafe_allow_html=True)

    rec_col1, rec_col2 = st.columns(2)
    with rec_col1:
        if st.button("⏺ Start REC" if not st.session_state.recording else "⏹ Stop REC"):
            st.session_state.recording = not st.session_state.recording
            st.rerun()
    with rec_col2:
        if st.button("📸 Snapshot"):
            st.session_state.snap_count += 1
            st.session_state["do_snapshot"] = True

    st.markdown("---")
    # Stats
    s1, s2 = st.columns(2)
    with s1:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-val">{st.session_state.snap_count}</div>
            <div class="stat-label">Snapshots</div>
        </div>""", unsafe_allow_html=True)
    with s2:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-val">{st.session_state.frame_count}</div>
            <div class="stat-label">Frames</div>
        </div>""", unsafe_allow_html=True)

    # Snapshot gallery
    if st.session_state.snapshots:
        st.markdown('<div class="gallery-title">Snapshot Gallery</div>', unsafe_allow_html=True)
        for snap_path, snap_time in reversed(st.session_state.snapshots[-6:]):
            if os.path.exists(snap_path):
                st.image(snap_path, caption=snap_time, use_container_width=True)

# ─── MAIN AREA ─────────────────────────────────────────────────────────────────
header_left, header_right = st.columns([3, 1])
with header_left:
    st.markdown(f"### {FILTER_EMOJIS[st.session_state.current_filter]} Live Feed — {FILTER_NAMES[st.session_state.current_filter]}")
with header_right:
    stop = st.button("■ Stop Camera", type="secondary")

feed_placeholder = st.empty()
fps_placeholder = st.empty()

# ─── VIDEO WRITER SETUP ────────────────────────────────────────────────────────
writer = None
fourcc = cv2.VideoWriter_fourcc(*'XVID')

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prev_time = time.time()

if not stop:
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera not accessible.")
            break

        frame = cv2.flip(frame, 1)
        frame, active_filter, mouth_val = process_frame(
            frame, st.session_state.current_filter, prev_points
        )
        st.session_state.current_filter = active_filter
        st.session_state.frame_count += 1

        # FPS
        curr = time.time()
        fps = 1 / (curr - prev_time + 0.0001)
        prev_time = curr

        # Recording
        if st.session_state.recording:
            if writer is None:
                writer = cv2.VideoWriter("output.avi", fourcc, 20, (640, 480))
            writer.write(frame)
        else:
            if writer is not None:
                writer.release()
                writer = None

        # Snapshot
        if st.session_state.get("do_snapshot"):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            snap_path = f"snapshots/snap_{ts}.png"
            cv2.imwrite(snap_path, frame)
            st.session_state.snapshots.append((snap_path, ts))
            st.session_state["do_snapshot"] = False

        # Overlay HUD on frame
        hud_frame = frame.copy()
        rec_text = "● REC" if st.session_state.recording else ""
        if rec_text:
            cv2.putText(hud_frame, rec_text, (frame.shape[1] - 90, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 220), 2)
        cv2.putText(hud_frame, f"FPS: {int(fps)}", (16, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 220, 80), 2)
        cv2.putText(hud_frame, f"FILTER: {FILTER_NAMES[st.session_state.current_filter].upper()}",
                    (16, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 180, 100), 1)

        # Display
        rgb_frame = cv2.cvtColor(hud_frame, cv2.COLOR_BGR2RGB)
        feed_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

cap.release()
if writer:
    writer.release()