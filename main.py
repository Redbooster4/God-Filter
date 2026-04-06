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
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500;600&family=Unbounded:wght@300;400;700&display=swap');
 
    /* ─── TOKENS ─────────────────────────────────────────── */
    :root {
        --bg-void:      #060609;
        --bg-deep:      #0a0a0f;
        --bg-surface:   #0e0e18;
        --bg-raised:    #141422;
        --bg-hover:     #1a1a2e;
 
        --border-dim:   #16162a;
        --border-base:  #20203a;
        --border-lit:   #2e2e52;
 
        --text-primary: #ede8df;
        --text-secondary:#9994aa;
        --text-muted:   #44405a;
        --text-ghost:   #2a2840;
 
        --accent:       #ff6b35;
        --accent-dim:   #cc4f1f;
        --accent-glow:  rgba(255, 107, 53, 0.15);
        --accent-ghost: rgba(255, 107, 53, 0.06);
 
        --warm:         #f7c59f;
        --cold:         #7b79ff;
        --danger:       #ff3355;
        --safe:         #39d98a;
 
        --radius-sm: 3px;
        --radius-md: 6px;
        --radius-lg: 10px;
 
        --font-display: 'Unbounded', sans-serif;
        --font-ui:      'Syne', sans-serif;
        --font-mono:    'JetBrains Mono', monospace;
 
        --transition: 0.18s cubic-bezier(0.4, 0, 0.2, 1);
        --transition-slow: 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    }
 
    /* ─── RESET & BASE ───────────────────────────────────── */
    *, *::before, *::after { box-sizing: border-box; }
 
    html, body, [class*="css"] {
        font-family: var(--font-ui);
        background-color: var(--bg-void);
        color: var(--text-primary);
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }
 
    .stApp {
        background: var(--bg-void);
    }
 
    /* Subtle scanline overlay on entire app */
    .stApp::before {
        content: '';
        position: fixed;
        inset: 0;
        background: repeating-linear-gradient(
            0deg,
            transparent,
            transparent 2px,
            rgba(0,0,0,0.04) 2px,
            rgba(0,0,0,0.04) 4px
        );
        pointer-events: none;
        z-index: 9999;
    }
 
    /* ─── SIDEBAR ─────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: var(--bg-surface) !important;
        border-right: 1px solid var(--border-base) !important;
        box-shadow: 4px 0 40px rgba(0,0,0,0.6);
    }
 
    [data-testid="stSidebar"] * {
        color: var(--text-primary) !important;
    }
 
    [data-testid="stSidebar"] .stMarkdown p {
        font-family: var(--font-mono);
        font-size: 0.68rem;
        color: var(--text-muted) !important;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
 
    /* Sidebar top accent line */
    [data-testid="stSidebar"]::before {
        content: '';
        display: block;
        height: 2px;
        background: linear-gradient(90deg, var(--accent), transparent);
        margin-bottom: 8px;
    }
 
    /* ─── TYPOGRAPHY ──────────────────────────────────────── */
    .god-title {
        font-family: var(--font-display);
        font-weight: 700;
        font-size: 2rem;
        letter-spacing: -2px;
        line-height: 1;
        background: linear-gradient(135deg,
            var(--accent) 0%,
            var(--warm) 50%,
            #ffffd1 100%
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 2px;
        position: relative;
    }
 
    /* Glitch flicker effect on title */
    .god-title::after {
        content: attr(data-text);
        position: absolute;
        left: 2px;
        top: 0;
        background: linear-gradient(135deg, #7b79ff, var(--accent));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        clip-path: polygon(0 30%, 100% 30%, 100% 50%, 0 50%);
        animation: glitch 6s infinite;
        opacity: 0.6;
    }
 
    @keyframes glitch {
        0%, 92%, 100% { opacity: 0; transform: translateX(0); }
        93%            { opacity: 0.6; transform: translateX(-3px); clip-path: polygon(0 20%, 100% 20%, 100% 40%, 0 40%); }
        95%            { opacity: 0.4; transform: translateX(3px); clip-path: polygon(0 60%, 100% 60%, 100% 80%, 0 80%); }
        97%            { opacity: 0; }
    }
 
    .god-sub {
        font-family: var(--font-mono);
        font-size: 0.62rem;
        color: var(--text-ghost);
        letter-spacing: 4px;
        text-transform: uppercase;
        margin-top: 6px;
        margin-bottom: 28px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
 
    .god-sub::before {
        content: '';
        display: inline-block;
        width: 20px;
        height: 1px;
        background: var(--accent-dim);
    }
 
    /* ─── BUTTONS ─────────────────────────────────────────── */
    .stButton > button {
        font-family: var(--font-mono) !important;
        font-weight: 500 !important;
        font-size: 0.68rem !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
        background: var(--bg-raised) !important;
        color: var(--text-secondary) !important;
        border: 1px solid var(--border-base) !important;
        border-radius: var(--radius-sm) !important;
        padding: 10px 16px !important;
        width: 100% !important;
        transition: all var(--transition) !important;
        position: relative !important;
        overflow: hidden !important;
    }
 
    /* Sweep shimmer on hover */
    .stButton > button::before {
        content: '';
        position: absolute;
        inset: 0;
        background: linear-gradient(90deg, transparent, rgba(255,107,53,0.08), transparent);
        transform: translateX(-100%);
        transition: transform 0.4s ease;
    }
 
    .stButton > button:hover::before {
        transform: translateX(100%);
    }
 
    .stButton > button:hover {
        background: var(--bg-hover) !important;
        border-color: var(--accent) !important;
        color: var(--accent) !important;
        box-shadow: 0 0 20px var(--accent-glow), inset 0 0 20px var(--accent-ghost) !important;
    }
 
    .stButton > button:active {
        transform: scale(0.98) !important;
    }
 
    /* ─── PILLS & BADGES ──────────────────────────────────── */
    .active-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: var(--accent);
        color: var(--bg-void);
        font-family: var(--font-mono);
        font-size: 0.62rem;
        font-weight: 600;
        letter-spacing: 3px;
        text-transform: uppercase;
        padding: 5px 14px;
        border-radius: var(--radius-sm);
        margin-bottom: 16px;
        box-shadow: 0 0 16px var(--accent-glow);
    }
 
    .active-pill::before {
        content: '▸';
        font-size: 0.7rem;
    }
 
    /* Recording badge */
    .rec-badge {
        display: inline-flex;
        align-items: center;
        gap: 10px;
        background: rgba(255,35,35,0.07);
        border: 1px solid rgba(255,35,35,0.3);
        color: #ff5566;
        font-family: var(--font-mono);
        font-size: 0.62rem;
        font-weight: 500;
        letter-spacing: 3px;
        text-transform: uppercase;
        padding: 7px 16px;
        border-radius: var(--radius-sm);
        box-shadow: 0 0 24px rgba(255,35,35,0.08);
        animation: pulse-border 2s ease-in-out infinite;
    }
 
    @keyframes pulse-border {
        0%, 100% { border-color: rgba(255,35,35,0.3); }
        50%       { border-color: rgba(255,35,35,0.7); box-shadow: 0 0 32px rgba(255,35,35,0.15); }
    }
 
    .rec-dot {
        width: 7px;
        height: 7px;
        background: var(--danger);
        border-radius: 50%;
        box-shadow: 0 0 8px var(--danger);
        animation: blink 1.1s ease-in-out infinite;
    }
 
    @keyframes blink {
        0%, 100% { opacity: 1; box-shadow: 0 0 8px var(--danger); }
        50%       { opacity: 0.15; box-shadow: none; }
    }
 
    /* Idle badge */
    .idle-badge {
        display: inline-flex;
        align-items: center;
        gap: 10px;
        background: rgba(57,217,138,0.05);
        border: 1px solid rgba(57,217,138,0.2);
        color: var(--safe);
        font-family: var(--font-mono);
        font-size: 0.62rem;
        font-weight: 500;
        letter-spacing: 3px;
        text-transform: uppercase;
        padding: 7px 16px;
        border-radius: var(--radius-sm);
    }
 
    /* ─── STATS ───────────────────────────────────────────── */
    .stat-box {
        background: var(--bg-raised);
        border: 1px solid var(--border-base);
        border-radius: var(--radius-md);
        padding: 16px 20px;
        text-align: center;
        transition: all var(--transition);
        position: relative;
        overflow: hidden;
    }
 
    .stat-box::after {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--accent-dim), transparent);
        opacity: 0;
        transition: opacity var(--transition);
    }
 
    .stat-box:hover {
        border-color: var(--border-lit);
        background: var(--bg-hover);
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.4), 0 0 24px var(--accent-ghost);
    }
 
    .stat-box:hover::after {
        opacity: 1;
    }
 
    .stat-val {
        font-family: var(--font-mono);
        font-size: 1.6rem;
        font-weight: 600;
        color: var(--accent);
        line-height: 1;
        letter-spacing: -1px;
        text-shadow: 0 0 20px var(--accent-glow);
    }
 
    .stat-label {
        font-family: var(--font-mono);
        font-size: 0.58rem;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-top: 6px;
    }
 
    /* ─── GALLERY ─────────────────────────────────────────── */
    .gallery-title {
        font-family: var(--font-mono);
        font-size: 0.58rem;
        letter-spacing: 5px;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 14px;
        margin-top: 28px;
        padding-top: 20px;
        border-top: 1px solid var(--border-dim);
        display: flex;
        align-items: center;
        gap: 12px;
    }
 
    .gallery-title::after {
        content: '';
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, var(--border-dim), transparent);
    }
 
    /* ─── FEED FRAME ──────────────────────────────────────── */
    .feed-container {
        border: 1px solid var(--border-base);
        border-radius: var(--radius-md);
        overflow: hidden;
        position: relative;
        box-shadow: 0 0 40px rgba(0,0,0,0.5), inset 0 0 60px rgba(0,0,0,0.3);
    }
 
    /* Corner brackets on feed */
    .feed-container::before,
    .feed-container::after {
        content: '';
        position: absolute;
        width: 16px;
        height: 16px;
        border-color: var(--accent);
        border-style: solid;
        z-index: 2;
    }
 
    .feed-container::before {
        top: 6px; left: 6px;
        border-width: 2px 0 0 2px;
    }
 
    .feed-container::after {
        bottom: 6px; right: 6px;
        border-width: 0 2px 2px 0;
    }
 
    /* ─── DIVIDERS ────────────────────────────────────────── */
    hr {
        border: none !important;
        border-top: 1px solid var(--border-dim) !important;
        margin: 24px 0 !important;
    }
 
    /* ─── FORM ELEMENTS ───────────────────────────────────── */
    [data-testid="stSelectbox"] label,
    [data-testid="stRadio"] label {
        font-family: var(--font-mono) !important;
        font-size: 0.62rem !important;
        letter-spacing: 3px !important;
        text-transform: uppercase !important;
        color: var(--text-muted) !important;
    }
 
    /* Selectbox dropdown */
    [data-testid="stSelectbox"] > div > div {
        background: var(--bg-raised) !important;
        border-color: var(--border-base) !important;
        border-radius: var(--radius-sm) !important;
        font-family: var(--font-mono) !important;
        font-size: 0.8rem !important;
        transition: border-color var(--transition) !important;
    }
 
    [data-testid="stSelectbox"] > div > div:hover {
        border-color: var(--accent-dim) !important;
    }
 
    /* Sliders */
    [data-testid="stSlider"] .stSlider > div > div > div {
        background: var(--accent) !important;
    }
 
    /* Text inputs */
    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea {
        background: var(--bg-raised) !important;
        border: 1px solid var(--border-base) !important;
        border-radius: var(--radius-sm) !important;
        color: var(--text-primary) !important;
        font-family: var(--font-mono) !important;
        font-size: 0.8rem !important;
        transition: border-color var(--transition), box-shadow var(--transition) !important;
    }
 
    [data-testid="stTextInput"] input:focus,
    [data-testid="stTextArea"] textarea:focus {
        border-color: var(--accent-dim) !important;
        box-shadow: 0 0 0 2px var(--accent-ghost) !important;
        outline: none !important;
    }
 
    /* ─── SCROLLBAR ───────────────────────────────────────── */
    ::-webkit-scrollbar             { width: 4px; height: 4px; }
    ::-webkit-scrollbar-track       { background: var(--bg-void); }
    ::-webkit-scrollbar-thumb       { background: var(--border-lit); border-radius: 2px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--accent-dim); }
 
    /* ─── EXPANDER ────────────────────────────────────────── */
    [data-testid="stExpander"] {
        background: var(--bg-raised) !important;
        border: 1px solid var(--border-base) !important;
        border-radius: var(--radius-md) !important;
    }
 
    [data-testid="stExpander"]:hover {
        border-color: var(--border-lit) !important;
    }
 
    /* ─── METRICS ─────────────────────────────────────────── */
    [data-testid="stMetric"] {
        background: var(--bg-raised);
        border: 1px solid var(--border-base);
        border-radius: var(--radius-md);
        padding: 16px !important;
        transition: all var(--transition);
    }
 
    [data-testid="stMetric"]:hover {
        border-color: var(--border-lit);
        box-shadow: 0 0 24px var(--accent-ghost);
    }
 
    [data-testid="stMetricValue"] {
        font-family: var(--font-mono) !important;
        color: var(--accent) !important;
        font-size: 1.5rem !important;
    }
 
    [data-testid="stMetricLabel"] {
        font-family: var(--font-mono) !important;
        font-size: 0.6rem !important;
        letter-spacing: 3px !important;
        text-transform: uppercase !important;
        color: var(--text-muted) !important;
    }
 
    /* ─── ALERTS / INFO ───────────────────────────────────── */
    [data-testid="stAlert"] {
        background: var(--bg-raised) !important;
        border-radius: var(--radius-md) !important;
        border-left-width: 3px !important;
        font-family: var(--font-mono) !important;
        font-size: 0.75rem !important;
    }
 
    /* ─── DATAFRAME / TABLE ───────────────────────────────── */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--border-base) !important;
        border-radius: var(--radius-md) !important;
        overflow: hidden;
    }
 
    /* ─── FADE IN ANIMATION FOR PAGE LOAD ────────────────── */
    .main .block-container {
        animation: fadeUp 0.5s ease both;
    }
 
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(12px); }
        to   { opacity: 1; transform: translateY(0); }
    }
 
    /* ─── UTILITY CLASSES ─────────────────────────────────── */
    .mono   { font-family: var(--font-mono); }
    .muted  { color: var(--text-muted); }
    .accent { color: var(--accent); }
 
    .tag {
        display: inline-block;
        font-family: var(--font-mono);
        font-size: 0.6rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        padding: 3px 10px;
        border: 1px solid var(--border-base);
        border-radius: var(--radius-sm);
        color: var(--text-muted);
    }
 
    .tag.accent {
        border-color: var(--accent-dim);
        color: var(--accent);
        background: var(--accent-ghost);
    }
 
    .section-header {
        font-family: var(--font-mono);
        font-size: 0.6rem;
        letter-spacing: 5px;
        text-transform: uppercase;
        color: var(--text-muted);
        padding-bottom: 10px;
        border-bottom: 1px solid var(--border-dim);
        margin-bottom: 16px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
 
    </style>
    """, unsafe_allow_html=True)
import cv2
import numpy as np
import mediapipe as mp
import streamlit as st
import os
import time
from datetime import datetime
from collections import deque

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
if "do_snapshot" not in st.session_state:
    st.session_state.do_snapshot = False

# Cooldown trackers (avoid rapid re-triggers)
if "smile_cooldown" not in st.session_state:
    st.session_state.smile_cooldown = 0
if "brow_cooldown" not in st.session_state:
    st.session_state.brow_cooldown = 0
if "palm_cooldown" not in st.session_state:
    st.session_state.palm_cooldown = 0

FILTER_NAMES  = ["Sunglasses", "Mustache", "Crown", "Horns"]
FILTER_EMOJIS = ["🕶️", "👨", "👑", "😈"]

os.makedirs("filters",   exist_ok=True)
os.makedirs("snapshots", exist_ok=True)

# ─── MEDIAPIPE LOADERS ─────────────────────────────────────────────────────────
@st.cache_resource
def load_face_mesh():
    mp_face_mesh = mp.solutions.face_mesh
    return mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

@st.cache_resource
def load_hands():
    mp_hands = mp.solutions.hands
    return mp_hands.Hands(
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
        "filters/horns.png",       # replaced dog_ears with horns
    ]
    return [cv2.imread(p, cv2.IMREAD_UNCHANGED) for p in paths]

face_mesh = load_face_mesh()
hands_detector = load_hands()
filters = load_filters()
prev_points = {}

# ─── HELPERS ───────────────────────────────────────────────────────────────────
def smooth_point(store, id_, x, y, alpha=0.3):
    """Exponential moving average smoothing."""
    if id_ not in store:
        store[id_] = (x, y)
    px, py = store[id_]
    sx = int(px * (1 - alpha) + x * alpha)
    sy = int(py * (1 - alpha) + y * alpha)
    store[id_] = (sx, sy)
    return sx, sy

def overlay(frame, img, x, y, w, h, angle=0):
    """Alpha-blend a PNG overlay onto frame at (x,y) with size (w,h)."""
    if img is None or w <= 0 or h <= 0:
        return frame
    img = cv2.resize(img, (w, h))
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    fh, fw = img.shape[:2]
    if angle != 0:
        M = cv2.getRotationMatrix2D((fw // 2, fh // 2), angle, 1)
        img = cv2.warpAffine(img, M, (fw, fh))
    # Bounds check
    if x < 0 or y < 0 or y + fh > frame.shape[0] or x + fw > frame.shape[1]:
        return frame
    b, g, r, a = cv2.split(img)
    mask = cv2.merge((a, a, a)) / 255.0
    roi  = frame[y:y + fh, x:x + fw].astype(np.float32)
    blended = roi * (1 - mask) + cv2.merge((b, g, r)).astype(np.float32) * mask
    frame[y:y + fh, x:x + fw] = blended.astype(np.uint8)
    return frame

# ─── SMILE DETECTION ───────────────────────────────────────────────────────────
# Mouth corner landmarks: 61 (left), 291 (right)
# Lip top/bottom: 13, 14  — used only to normalize by face width
def detect_smile(lm, w, h, face_width_px):
    """
    Returns True if mouth corners are pulled wide enough relative to face width.
    Threshold tuned for genuine smile, not neutral open mouth.
    """
    left_corner  = lm[61]
    right_corner = lm[291]
    mouth_width  = abs(right_corner.x - left_corner.x) * w
    ratio = mouth_width / (face_width_px + 1e-6)
    # ratio > 0.42 is a relaxed smile; > 0.48 is a big grin
    return ratio > 0.44

# ─── EYEBROW RAISE DETECTION ───────────────────────────────────────────────────
# Eyebrow top: 70 (left), 300 (right)
# Eye center:  159 (left upper lid), 386 (right upper lid)
def detect_eyebrow_raise(lm, h):
    """
    Returns True when BOTH eyebrows are significantly above their eye lids.
    Normalized by face height so it works at any distance from camera.
    """
    left_brow_y  = lm[70].y  * h
    right_brow_y = lm[300].y * h
    left_eye_y   = lm[159].y * h
    right_eye_y  = lm[386].y * h

    left_gap  = left_eye_y  - left_brow_y
    right_gap = right_eye_y - right_brow_y

    face_height = abs(lm[10].y - lm[152].y) * h  # forehead to chin
    # Gap > 8% of face height = raised brows
    threshold = face_height * 0.08
    return (left_gap > threshold) and (right_gap > threshold)

# ─── OPEN PALM DETECTION ───────────────────────────────────────────────────────
# All 5 fingertips (4,8,12,16,20) must be above their MCP knuckles (2,5,9,13,17)
# AND fingers must be spread (not a fist)
def detect_open_palm(hand_landmarks, frame_h):
    """
    Returns True when a flat open palm is shown.
    Checks all 5 fingertips are extended above their respective knuckles.
    """
    lm = hand_landmarks.landmark
    tips   = [4, 8, 12, 16, 20]   # fingertip ids
    knuckles = [2, 5,  9, 13, 17] # MCP/base knuckle ids

    extended = 0
    for tip, knuckle in zip(tips, knuckles):
        # y decreases upward in image coords — tip.y < knuckle.y means extended
        if tip == 4:
            # Thumb: compare x instead of y (horizontal extension)
            if abs(lm[tip].x - lm[knuckle].x) > 0.06:
                extended += 1
        else:
            if lm[tip].y < lm[knuckle].y - 0.04:
                extended += 1

    return extended >= 4   # at least 4/5 fingers extended = open palm

# ─── MAIN FRAME PROCESSOR ──────────────────────────────────────────────────────
def process_frame(frame, current_filter, store, frame_idx):
    """
    Returns: (processed_frame, active_filter, trigger_info_dict)
    trigger_info carries booleans for HUD display.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    info = {"smile": False, "brow": False, "palm": False}

    # ── Hand detection ──────────────────────────────────────
    hand_results = hands_detector.process(rgb)
    if hand_results.multi_hand_landmarks:
        for hand_lm in hand_results.multi_hand_landmarks:
            if detect_open_palm(hand_lm, frame.shape[0]):
                info["palm"] = True

    # ── Face detection ──────────────────────────────────────
    face_results = face_mesh.process(rgb)

    if face_results.multi_face_landmarks:
        for face in face_results.multi_face_landmarks:
            lm = face.landmark
            h, w, _ = frame.shape

            # Eye corner landmarks for face width + angle
            left_eye  = lm[33]
            right_eye = lm[263]
            x1, y1 = int(left_eye.x  * w), int(left_eye.y  * h)
            x2, y2 = int(right_eye.x * w), int(right_eye.y * h)
            x1, y1 = smooth_point(store, 1, x1, y1)
            x2, y2 = smooth_point(store, 2, x2, y2)

            eye_dist     = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            face_width   = int(eye_dist * 1.5)
            angle        = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            ox = cx - face_width // 2
            oy = cy - face_width // 4

            # Nose
            nose = lm[1]
            nx, ny = smooth_point(store, 3, int(nose.x * w), int(nose.y * h))

            # Forehead
            head = lm[10]
            hx, hy = smooth_point(store, 4, int(head.x * w), int(head.y * h))

            # ── SMILE → force sunglasses ───────────────────
            if detect_smile(lm, w, h, face_width):
                info["smile"] = True
                if st.session_state.smile_cooldown < frame_idx:
                    current_filter = 0                          # sunglasses
                    st.session_state.smile_cooldown = frame_idx + 30  # 30 frame lock

            # ── EYEBROW RAISE → force crown ───────────────
            elif detect_eyebrow_raise(lm, h):
                info["brow"] = True
                if st.session_state.brow_cooldown < frame_idx:
                    current_filter = 2                          # crown
                    st.session_state.brow_cooldown = frame_idx + 30

            # ── DRAW ACTIVE FILTER ─────────────────────────
            if current_filter == 0:   # Sunglasses
                frame = overlay(frame, filters[0],
                                ox, oy, face_width, int(face_width * 0.5), angle)

            elif current_filter == 1: # Mustache
                frame = overlay(frame, filters[1],
                                nx - face_width // 4, ny,
                                face_width // 2, face_width // 4, 0)

            elif current_filter == 2: # Crown
                frame = overlay(frame, filters[2],
                                hx - face_width // 2, hy - face_width // 2,
                                face_width, face_width // 2, 0)

            elif current_filter == 3: # Horns
                frame = overlay(frame, filters[3],
                                hx - face_width // 2, hy - face_width,
                                face_width, face_width // 2, 0)

    return frame, current_filter, info

# ─── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="god-title" data-text="GOD FILTER">GOD<br>FILTER</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="god-sub">Face AR · v2.0</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="gallery-title" style="border-top:none;padding-top:0;margin-top:0">Active Filter</div>',
                unsafe_allow_html=True)

    active_name  = FILTER_NAMES[st.session_state.current_filter]
    active_emoji = FILTER_EMOJIS[st.session_state.current_filter]
    st.markdown(f'<div class="active-pill">{active_emoji} {active_name}</div>',
                unsafe_allow_html=True)

    cols = st.columns(2)
    for i, (name, emoji) in enumerate(zip(FILTER_NAMES, FILTER_EMOJIS)):
        with cols[i % 2]:
            if st.button(f"{emoji} {name}", key=f"filter_{i}"):
                st.session_state.current_filter = i

    st.markdown("---")
    st.markdown('<div class="gallery-title" style="border-top:none;padding-top:0;margin-top:0">Gesture Triggers</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:var(--font-mono,monospace);font-size:0.65rem;
                color:#555577;line-height:2;letter-spacing:1px;">
        😁 &nbsp;SMILE → Sunglasses<br>
        🤨 &nbsp;RAISE BROWS → Crown<br>
        🤚 &nbsp;OPEN PALM → Snapshot
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="gallery-title" style="border-top:none;padding-top:0;margin-top:0">Recording</div>',
                unsafe_allow_html=True)

    if st.session_state.recording:
        st.markdown('<div class="rec-badge"><div class="rec-dot"></div>REC ACTIVE</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="idle-badge">● IDLE</div>', unsafe_allow_html=True)

    rec_col1, rec_col2 = st.columns(2)
    with rec_col1:
        label = "⏹ Stop REC" if st.session_state.recording else "⏺ Start REC"
        if st.button(label):
            st.session_state.recording = not st.session_state.recording
            st.rerun()
    with rec_col2:
        if st.button("📸 Snapshot"):
            st.session_state.do_snapshot = True

    st.markdown("---")
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

    if st.session_state.snapshots:
        st.markdown('<div class="gallery-title">Snapshot Gallery</div>', unsafe_allow_html=True)
        for snap_path, snap_time in reversed(st.session_state.snapshots[-6:]):
            if os.path.exists(snap_path):
                st.image(snap_path, caption=snap_time, use_container_width=True)

# ─── MAIN AREA ─────────────────────────────────────────────────────────────────
header_left, header_right = st.columns([3, 1])
with header_left:
    st.markdown(
        f"### {FILTER_EMOJIS[st.session_state.current_filter]} "
        f"Live Feed — {FILTER_NAMES[st.session_state.current_filter]}"
    )
with header_right:
    stop = st.button("■ Stop Camera", type="secondary")

feed_placeholder = st.empty()

# ─── VIDEO WRITER SETUP ────────────────────────────────────────────────────────
writer = None
fourcc = cv2.VideoWriter_fourcc(*'XVID')

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prev_time  = time.time()
frame_idx  = 0   # local counter, separate from session state frame_count

if not stop:
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera not accessible.")
            break

        frame     = cv2.flip(frame, 1)
        frame_idx += 1

        frame, active_filter, info = process_frame(
            frame,
            st.session_state.current_filter,
            prev_points,
            frame_idx,
        )
        st.session_state.current_filter  = active_filter
        st.session_state.frame_count    += 1

        # FPS
        curr      = time.time()
        fps       = 1 / (curr - prev_time + 1e-6)
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

        # ── Snapshot: button OR open palm ──────────────────
        palm_triggered = (
            info["palm"]
            and st.session_state.palm_cooldown < frame_idx
        )
        if st.session_state.do_snapshot or palm_triggered:
            ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
            snap_path = f"snapshots/snap_{ts}.png"
            cv2.imwrite(snap_path, frame)
            st.session_state.snapshots.append((snap_path, ts))
            st.session_state.snap_count        += 1
            st.session_state.do_snapshot        = False
            # 90 frame cooldown (~3 s at 30fps) prevents burst snaps
            st.session_state.palm_cooldown      = frame_idx + 90

        # ── HUD overlay ────────────────────────────────────
        hud = frame.copy()
        cv2.putText(hud, f"FPS: {int(fps)}",
                    (16, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 220, 80), 2)
        cv2.putText(hud, f"FILTER: {FILTER_NAMES[active_filter].upper()}",
                    (16, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 180, 100), 1)

        if st.session_state.recording:
            cv2.putText(hud, "● REC",
                        (frame.shape[1] - 90, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 220), 2)

        # Live gesture indicators on HUD
        if info["smile"]:
            cv2.putText(hud, "SMILE :)",
                        (16, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 220, 180), 1)
        if info["brow"]:
            cv2.putText(hud, "BROWS UP",
                        (16, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 80, 220), 1)
        if info["palm"]:
            cv2.putText(hud, "PALM  → SNAP",
                        (16, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 180, 255), 1)

        rgb_frame = cv2.cvtColor(hud, cv2.COLOR_BGR2RGB)
        feed_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

cap.release()
if writer:
    writer.release()