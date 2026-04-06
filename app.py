"""
app.py  —  GOD FILTER v3.0
─────────────────────────────────────────────────────────
Features
  [Original]  Face filters (sunglasses, mustache, crown, horns)
  [Gesture]   Smile → sunglasses  |  Brow raise → crown
  [Feature 2] Head pose estimation (yaw / pitch / roll)
  [Feature 3] Blink rate + EAR + fatigue level
  [Feature 6] Face identity lock (enroll + recognize)
  [Feature 7] Custom gesture classifier (train in-app)

Run:  streamlit run app.py
"""

import os, time
import cv2
import numpy as np
import mediapipe as mp
import streamlit as st
from datetime import datetime

# ── Local modules ────────────────────────────────────────
from head_pose          import estimate_pose, draw_pose_hud
from blink_counter      import BlinkTracker, draw_blink_hud
from face_id            import FaceIDSystem, draw_id_hud
from gesture_classifier import (
    GestureCollector, GestureClassifier,
    GESTURE_LABELS, draw_gesture_hud,
)

# ── Inline CSS (paste your improved_styles.py inject here) ──
# from styles import inject_styles; inject_styles()

st.set_page_config(page_title="GOD FILTER", layout="wide")

# ─── SESSION STATE ────────────────────────────────────────
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
_init("enroll_name",     "User")
_init("enroll_count",    0)
_init("collecting",      False)   # gesture data collection active
_init("collect_label",   GESTURE_LABELS[0])
_init("collect_counts",  {g: 0 for g in GESTURE_LABELS})
_init("model_trained",   False)
_init("authorized_only", False)
_init("show_pose",       True)
_init("show_blink",      True)
_init("show_id",         True)
_init("show_gesture",    True)

FILTER_NAMES  = ["Sunglasses", "Mustache", "Crown", "Horns"]
FILTER_EMOJIS = ["🕶️", "👨", "👑", "😈"]

os.makedirs("filters",   exist_ok=True)
os.makedirs("snapshots", exist_ok=True)
os.makedirs("models",    exist_ok=True)

# ─── CACHED RESOURCES ────────────────────────────────────
@st.cache_resource
def load_face_mesh():
    return mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    )

@st.cache_resource
def load_hands():
    return mp.solutions.hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.6, min_tracking_confidence=0.5,
    )

@st.cache_resource
def load_filters():
    paths = [
        "filters/sunglasses.png",
        "filters/mustache.png",
        "filters/crown.png"
    ]

    imgs = []

    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)

        if img is None:
            print(f"Missing filter: {p}")
            img = np.zeros((10,10,4),dtype=np.uint8)

        imgs.append(img)

    return imgs

@st.cache_resource
def load_face_id():
    return FaceIDSystem()

@st.cache_resource
def load_gesture_classifier():
    gc = GestureClassifier(smooth_window=7)
    gc.load()   # loads saved model if exists
    return gc

@st.cache_resource
def load_gesture_collector():
    return GestureCollector(min_samples_per_class=80)

face_mesh   = load_face_mesh()
hands_det   = load_hands()
filters     = load_filters()
face_id_sys = load_face_id()
g_clf       = load_gesture_classifier()
g_col       = load_gesture_collector()
blink_tracker = BlinkTracker()
prev_points   = {}

# ─── HELPERS ─────────────────────────────────────────────
def smooth_point(store, id_, x, y, alpha=0.3):
    if id_ not in store:
        store[id_] = (x, y)
    px, py = store[id_]
    sx = int(px * (1-alpha) + x * alpha)
    sy = int(py * (1-alpha) + y * alpha)
    store[id_] = (sx, sy)
    return sx, sy

def overlay(frame, img, x, y, w, h, angle=0):
    if img is None or w <= 0 or h <= 0:
        return frame
    img = cv2.resize(img, (w, h))
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    fh, fw = img.shape[:2]
    if angle != 0:
        M = cv2.getRotationMatrix2D((fw//2, fh//2), angle, 1)
        img = cv2.warpAffine(img, M, (fw, fh))
    if x < 0 or y < 0 or y+fh > frame.shape[0] or x+fw > frame.shape[1]:
        return frame
    b, g, r, a = cv2.split(img)
    mask = cv2.merge((a,a,a)) / 255.0
    roi  = frame[y:y+fh, x:x+fw].astype(np.float32)
    frame[y:y+fh, x:x+fw] = (roi*(1-mask) + cv2.merge((b,g,r)).astype(np.float32)*mask).astype(np.uint8)
    return frame

def detect_smile(lm, w, h, face_width):
    mw = abs(lm[291].x - lm[61].x) * w
    return (mw / (face_width + 1e-6)) > 0.44

def detect_eyebrow_raise(lm, h):
    lby = lm[70].y  * h;  rby = lm[300].y * h
    ley = lm[159].y * h;  rey = lm[386].y * h
    fh  = abs(lm[10].y - lm[152].y) * h
    return ((ley - lby) > fh*0.08) and ((rey - rby) > fh*0.08)

def detect_open_palm_rules(hand_lm):
    lm = hand_lm.landmark
    tips, knuckles = [4,8,12,16,20], [2,5,9,13,17]
    ext = 0
    for tip, knk in zip(tips, knuckles):
        if tip == 4:
            ext += abs(lm[tip].x - lm[knk].x) > 0.06
        else:
            ext += lm[tip].y < lm[knk].y - 0.04
    return ext >= 4

# ─── FRAME PROCESSOR ─────────────────────────────────────
def process_frame(frame, current_filter, store, fidx):
    rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame.shape[:2]
    info = {
        "smile": False, "brow": False, "palm": False,
        "yaw": 0.0, "pitch": 0.0, "roll": 0.0,
        "blink": False, "ear": 1.0, "bpm": 0,
        "face_name": "Unknown", "face_conf": 0.0, "authorized": False,
        "gesture": "Unknown", "g_conf": 0.0, "g_action": None,
    }

    # ── Hands ────────────────────────────────────────────
    hand_results = hands_det.process(rgb)
    hand_lm_list = hand_results.multi_hand_landmarks or []

    for hand_lm in hand_lm_list:
        # Rule-based open palm
        if detect_open_palm_rules(hand_lm):
            info["palm"] = True

        # Trained gesture classifier
        if g_clf.is_trained:
            gesture, g_conf, g_action = g_clf.predict(hand_lm)
            info["gesture"]  = gesture
            info["g_conf"]   = g_conf
            info["g_action"] = g_action

        # Collect samples if collection mode is on
        if st.session_state.collecting:
            label = st.session_state.collect_label
            cnt, _ = g_col.collect(hand_lm, label)
            st.session_state.collect_counts[label] = cnt

    # ── Face mesh ────────────────────────────────────────
    face_results = face_mesh.process(rgb)

    if face_results.multi_face_landmarks:
        for face in face_results.multi_face_landmarks:
            lm = face.landmark

            # Eye geometry for filter placement
            x1,y1 = smooth_point(store,1, int(lm[33].x*w),  int(lm[33].y*h))
            x2,y2 = smooth_point(store,2, int(lm[263].x*w), int(lm[263].y*h))
            eye_dist   = np.sqrt((x2-x1)**2+(y2-y1)**2)
            face_width = int(eye_dist * 1.5)
            angle      = np.degrees(np.arctan2(y2-y1, x2-x1))
            cx,cy      = (x1+x2)//2, (y1+y2)//2
            ox,oy      = cx - face_width//2, cy - face_width//4

            nx,ny = smooth_point(store,3, int(lm[1].x*w),  int(lm[1].y*h))
            hx,hy = smooth_point(store,4, int(lm[10].x*w), int(lm[10].y*h))

            # ── FEATURE 2: Head Pose ──────────────────────
            yaw, pitch, roll = estimate_pose(lm, w, h)
            info["yaw"], info["pitch"], info["roll"] = yaw, pitch, roll

            # ── FEATURE 3: Blink ─────────────────────────
            blink_now, ear, bpm = blink_tracker.update(lm, w, h)
            info["blink"] = blink_now
            info["ear"]   = ear
            info["bpm"]   = bpm

            # ── FEATURE 6: Face ID ────────────────────────
            if st.session_state.show_id:
                name, conf = face_id_sys.identify(frame, lm)
                auth       = name in face_id_sys.enrolled_names() and conf >= 0.55
                info["face_name"]  = name
                info["face_conf"]  = conf
                info["authorized"] = auth

            # ── Smile / Brow gesture triggers ─────────────
            if detect_smile(lm, w, h, face_width):
                info["smile"] = True
                if st.session_state.smile_cd < fidx:
                    current_filter = 0
                    st.session_state.smile_cd = fidx + 30

            elif detect_eyebrow_raise(lm, h):
                info["brow"] = True
                if st.session_state.brow_cd < fidx:
                    current_filter = 2
                    st.session_state.brow_cd = fidx + 30

            # ── Draw filter ───────────────────────────────
            if current_filter == 0:
                frame = overlay(frame, filters[0], ox, oy, face_width, int(face_width*0.5), angle)
            elif current_filter == 1:
                frame = overlay(frame, filters[1], nx-face_width//4, ny, face_width//2, face_width//4, 0)
            elif current_filter == 2:
                frame = overlay(frame, filters[2], hx-face_width//2, hy-face_width//2, face_width, face_width//2, 0)
            elif current_filter == 3:
                frame = overlay(frame, filters[3], hx-face_width//2, hy-face_width, face_width, face_width//2, 0)

    return frame, current_filter, info

# ─── SIDEBAR ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="god-title">GOD<br>FILTER</div>', unsafe_allow_html=True)
    st.markdown('<div class="god-sub">Face AR · v3.0</div>', unsafe_allow_html=True)

    # ── Filters ───────────────────────────────────────────
    st.markdown("---")
    st.markdown("**FILTERS**")
    active_name = FILTER_NAMES[st.session_state.current_filter]
    st.markdown(f'<div class="active-pill">{FILTER_EMOJIS[st.session_state.current_filter]} {active_name}</div>',
                unsafe_allow_html=True)
    cols = st.columns(2)
    for i,(name,emoji) in enumerate(zip(FILTER_NAMES, FILTER_EMOJIS)):
        with cols[i%2]:
            if st.button(f"{emoji} {name}", key=f"f{i}"):
                st.session_state.current_filter = i

    # ── HUD toggles ───────────────────────────────────────
    st.markdown("---")
    st.markdown("**HUD OVERLAYS**")
    st.session_state.show_pose    = st.checkbox("Head Pose",    value=st.session_state.show_pose)
    st.session_state.show_blink   = st.checkbox("Blink / EAR", value=st.session_state.show_blink)
    st.session_state.show_id      = st.checkbox("Face ID",      value=st.session_state.show_id)
    st.session_state.show_gesture = st.checkbox("Gesture",      value=st.session_state.show_gesture)

    # ── Feature 6: Face ID enroll ─────────────────────────
    st.markdown("---")
    st.markdown("**FACE IDENTITY**")
    enrolled = face_id_sys.enrolled_names()
    if enrolled:
        st.markdown(f"Enrolled: `{'`, `'.join(enrolled)}`")
    st.session_state.enroll_name = st.text_input("Name to enroll", st.session_state.enroll_name)
    ecol1, ecol2 = st.columns(2)
    with ecol1:
        if st.button("📸 Enroll face"):
            st.session_state["do_enroll"] = True
    with ecol2:
        if st.button("🗑️ Clear IDs"):
            face_id_sys.clear()
            st.rerun()
    enroll_status = st.empty()
    st.session_state.authorized_only = st.checkbox(
        "Block unauthorized", value=st.session_state.authorized_only
    )

    # ── Feature 7: Gesture trainer ───────────────────────
    st.markdown("---")
    st.markdown("**GESTURE TRAINER**")

    collect_label = st.selectbox(
        "Gesture to collect", GESTURE_LABELS,
        index=GESTURE_LABELS.index(st.session_state.collect_label)
    )
    st.session_state.collect_label = collect_label

    counts = st.session_state.collect_counts
    for g in GESTURE_LABELS:
        c = counts.get(g, 0)
        bar = "█" * min(c//8, 10)
        color = "🟢" if c >= 80 else "🟡" if c >= 40 else "🔴"
        st.markdown(f"{color} `{g:<12}` {bar} {c}/80", unsafe_allow_html=False)

    gcol1, gcol2 = st.columns(2)
    with gcol1:
        label = "⏹ Stop" if st.session_state.collecting else "⏺ Collect"
        if st.button(label):
            st.session_state.collecting = not st.session_state.collecting
    with gcol2:
        train_disabled = not g_col.is_ready()
        if st.button("🧠 Train", disabled=train_disabled):
            with st.spinner("Training classifier…"):
                acc = g_clf.train(g_col, verbose=True)
                g_clf.save()
                st.session_state.model_trained = True
            st.success(f"Trained! CV accuracy: {acc:.1%}")

    if g_clf.is_trained:
        st.markdown("✅ Model loaded & ready")
    else:
        st.markdown("⚠️ No model — collect & train first")

    # ── Recording ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("**RECORDING**")
    if st.session_state.recording:
        st.markdown('<div class="rec-badge"><div class="rec-dot"></div>REC ACTIVE</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="idle-badge">● IDLE</div>', unsafe_allow_html=True)

    rc1, rc2 = st.columns(2)
    with rc1:
        lbl = "⏹ Stop" if st.session_state.recording else "⏺ REC"
        if st.button(lbl):
            st.session_state.recording = not st.session_state.recording
            st.rerun()
    with rc2:
        if st.button("📸 Snap"):
            st.session_state.do_snapshot = True

    # ── Stats ─────────────────────────────────────────────
    st.markdown("---")
    s1, s2 = st.columns(2)
    with s1:
        st.markdown(f'<div class="stat-box"><div class="stat-val">{st.session_state.snap_count}</div><div class="stat-label">Snaps</div></div>',
                    unsafe_allow_html=True)
    with s2:
        st.markdown(f'<div class="stat-box"><div class="stat-val">{st.session_state.frame_count}</div><div class="stat-label">Frames</div></div>',
                    unsafe_allow_html=True)

    # ── Snapshot gallery ──────────────────────────────────
    if st.session_state.snapshots:
        st.markdown('<div class="gallery-title">Gallery</div>', unsafe_allow_html=True)
        for snap_path, snap_time in reversed(st.session_state.snapshots[-6:]):
            if os.path.exists(snap_path):
                st.image(snap_path, caption=snap_time, use_container_width=True)

# ─── MAIN FEED ───────────────────────────────────────────
h_left, h_right = st.columns([3,1])
with h_left:
    st.markdown(f"### {FILTER_EMOJIS[st.session_state.current_filter]} Live Feed — {FILTER_NAMES[st.session_state.current_filter]}")
with h_right:
    stop = st.button("■ Stop Camera", type="secondary")

feed_ph = st.empty()

# Collection overlay indicator
collect_ph = st.empty()

# ─── VIDEO WRITER ────────────────────────────────────────
writer = None
fourcc = cv2.VideoWriter_fourcc(*'XVID')
cap    = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prev_time = time.time()
fidx      = 0

if not stop:
    while cap.isOpened() and not stop:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera not accessible.")
            break
        if stop:
            break

        frame = cv2.flip(frame, 1)
        fidx += 1
        st.session_state.frame_count += 1

        frame, active_filter, info = process_frame(
            frame, st.session_state.current_filter, prev_points, fidx
        )
        st.session_state.current_filter = active_filter

        # ── FPS ───────────────────────────────────────────
        curr = time.time()
        fps  = 1 / (curr - prev_time + 1e-6)
        prev_time = curr

        # ── Recording ────────────────────────────────────
        if st.session_state.recording:
            if writer is None:
                writer = cv2.VideoWriter("output.avi", fourcc, 20, (640,480))
            writer.write(frame)
        elif writer:
            writer.release(); writer = None

        # ── Enroll face ───────────────────────────────────
        if st.session_state.get("do_enroll"):
            ok, msg = face_id_sys.enroll(
                frame, st.session_state.enroll_name
            )
            st.session_state.enroll_count += 1
            st.session_state["do_enroll"] = False

        # ── Snapshot: button / palm / gesture ─────────────
        palm_snap    = info["palm"]    and st.session_state.palm_cd < fidx
        gesture_snap = info["g_action"] == "snapshot" and st.session_state.gesture_cd < fidx

        if st.session_state.do_snapshot or palm_snap or gesture_snap:
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"snapshots/snap_{ts}.png"
            cv2.imwrite(path, frame)
            st.session_state.snapshots.append((path, ts))
            st.session_state.snap_count      += 1
            st.session_state.do_snapshot      = False
            st.session_state.palm_cd          = fidx + 90
            st.session_state.gesture_cd       = fidx + 90

        # ── Gesture actions ───────────────────────────────
        if info["g_action"] == "filter_next" and st.session_state.gesture_cd < fidx:
            st.session_state.current_filter = (st.session_state.current_filter + 1) % 4
            st.session_state.gesture_cd = fidx + 60
        elif info["g_action"] == "filter_prev" and st.session_state.gesture_cd < fidx:
            st.session_state.current_filter = (st.session_state.current_filter - 1) % 4
            st.session_state.gesture_cd = fidx + 60
        elif info["g_action"] == "toggle_rec" and st.session_state.gesture_cd < fidx:
            st.session_state.recording = not st.session_state.recording
            st.session_state.gesture_cd = fidx + 90

        # ── HUD ───────────────────────────────────────────
        hud = frame.copy()

        # Base info
        cv2.putText(hud, f"FPS: {int(fps)}", (16,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80,220,80), 1)
        cv2.putText(hud, f"FILTER: {FILTER_NAMES[active_filter].upper()}", (16,55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,180,100), 1)

        # REC indicator
        if st.session_state.recording:
            cv2.putText(hud, "● REC", (frame.shape[1]-90, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,220), 2)

        # Collection indicator
        if st.session_state.collecting:
            lbl = st.session_state.collect_label
            cnt = st.session_state.collect_counts.get(lbl, 0)
            cv2.putText(hud, f"COLLECTING: {lbl.upper()} [{cnt}/80]",
                        (16, frame.shape[0]-16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,100,80), 2)

        # Feature 2: Head pose
        if st.session_state.show_pose:
            draw_pose_hud(hud, info["yaw"], info["pitch"], info["roll"], x=16, y=90)

        # Feature 3: Blink
        if st.session_state.show_blink:
            draw_blink_hud(hud, blink_tracker, info["bpm"], x=16, y=165)

        # Feature 6: Face ID
        if st.session_state.show_id:
            draw_id_hud(hud, info["face_name"], info["face_conf"],
                        info["authorized"], x=16, y=255)
            # Block feed if unauthorized
            if st.session_state.authorized_only and not info["authorized"]:
                overlay_block = np.zeros_like(hud)
                cv2.putText(overlay_block, "ACCESS DENIED",
                            (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,200), 3)
                hud = cv2.addWeighted(hud, 0.3, overlay_block, 0.7, 0)

        # Feature 7: Gesture
        if st.session_state.show_gesture:
            draw_gesture_hud(hud, info["gesture"], info["g_conf"],
                             info["g_action"], x=16, y=340)

        # Gesture trigger flash
        if info["smile"]:
            cv2.putText(hud, "SMILE → SUNGLASSES", (200,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80,255,180), 1)
        if info["brow"]:
            cv2.putText(hud, "BROWS UP → CROWN",   (200,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,80,255), 1)
        if info["palm"]:
            cv2.putText(hud, "PALM → SNAP",         (200,55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80,180,255), 1)

        rgb = cv2.cvtColor(hud, cv2.COLOR_BGR2RGB)
        feed_ph.image(rgb, channels="RGB", use_container_width=True)

cap.release()
if writer: writer.release()
