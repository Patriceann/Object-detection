import streamlit as st
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO
import av
import cv2
import numpy as np
from datetime import datetime
import os
import threading

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Live Object Detection & Tracing", layout="wide")

# ── Cache model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ── Shared state (thread-safe) ────────────────────────────────────────────────
lock = threading.Lock()
shared_state = {
    "counts": {},          # {label: count}
    "alerts": [],          # list of triggered alert messages
    "save_next": False,    # flag: save the next annotated frame
    "last_saved": None,    # filename of last saved frame
}

# ── Saved frames folder ───────────────────────────────────────────────────────
SAVE_DIR = "saved_frames"
os.makedirs(SAVE_DIR, exist_ok=True)

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("📸 Live Object Detection & Tracing")
st.write("Point your camera at objects to identify them in real-time.")

col1, col2 = st.columns([2, 1])

with col2:
    st.subheader("⚙️ Settings")
    conf_threshold = st.slider("Confidence threshold", 0.1, 1.0, 0.5, 0.05)

    st.subheader("🔔 Alert Triggers")
    st.caption("Trigger a visual alert when these objects are detected:")
    alert_person   = st.checkbox("Person",     value=True)
    alert_phone    = st.checkbox("Cell phone", value=False)
    alert_bottle   = st.checkbox("Bottle",     value=False)
    alert_backpack = st.checkbox("Backpack",   value=False)
    alert_laptop   = st.checkbox("Laptop",     value=False)

    custom_alert = st.text_input("Custom object label (optional)", placeholder="e.g. chair")

    # Build alert set from checkboxes
    alert_objects = set()
    if alert_person:   alert_objects.add("person")
    if alert_phone:    alert_objects.add("cell phone")
    if alert_bottle:   alert_objects.add("bottle")
    if alert_backpack: alert_objects.add("backpack")
    if alert_laptop:   alert_objects.add("laptop")
    if custom_alert.strip():
        alert_objects.add(custom_alert.strip().lower())

    st.subheader("💾 Save Frame")
    if st.button("📷 Capture current frame"):
        with lock:
            shared_state["save_next"] = True
        st.success("Frame will be saved on the next detection tick!")

    with lock:
        last = shared_state["last_saved"]
    if last:
        st.info(f"Last saved: `{last}`")

# ── Video frame callback ──────────────────────────────────────────────────────
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    # Run YOLOv8 tracking
    results = model.track(
        img,
        persist=True,
        conf=conf_threshold,
        verbose=False,
    )

    # ── Count objects ─────────────────────────────────────────────────────────
    counts = {}
    triggered_alerts = []

    if results[0].boxes is not None:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            label  = model.names[cls_id].lower()
            counts[label] = counts.get(label, 0) + 1

            # Alert check
            if label in alert_objects:
                msg = f"⚠️ ALERT: '{label}' detected!"
                if msg not in triggered_alerts:
                    triggered_alerts.append(msg)

    # ── Annotate frame ────────────────────────────────────────────────────────
    annotated_frame = results[0].plot()

    # Overlay count text on frame
    y_offset = 30
    for lbl, cnt in counts.items():
        text = f"{lbl}: {cnt}"
        cv2.putText(
            annotated_frame, text,
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (0, 255, 0), 2, cv2.LINE_AA,
        )
        y_offset += 28

    # Alert overlay banner (red bar at top if something triggered)
    if triggered_alerts:
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (0, 0), (annotated_frame.shape[1], 50), (0, 0, 200), -1)
        cv2.addWeighted(overlay, 0.5, annotated_frame, 0.5, 0, annotated_frame)
        cv2.putText(
            annotated_frame,
            " | ".join([a.replace("⚠️ ALERT: ", "") for a in triggered_alerts]),
            (10, 33),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65,
            (255, 255, 255), 2, cv2.LINE_AA,
        )

    # ── Save frame if requested ───────────────────────────────────────────────
    with lock:
        do_save = shared_state["save_next"]

    if do_save:
        filename = os.path.join(
            SAVE_DIR,
            f"frame_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
        )
        cv2.imwrite(filename, annotated_frame)
        with lock:
            shared_state["save_next"]  = False
            shared_state["last_saved"] = filename

    # ── Push state back ───────────────────────────────────────────────────────
    with lock:
        shared_state["counts"]  = counts
        shared_state["alerts"]  = triggered_alerts

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# ── WebRTC streamer ───────────────────────────────────────────────────────────
with col1:
    webrtc_streamer(
        key="object-detection",
        video_frame_callback=video_frame_callback,
        async_processing=True,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={"video": True, "audio": False},
    )

# ── Live stats panel (auto-refresh trick with empty placeholder) ──────────────
st.divider()
stat_col1, stat_col2 = st.columns(2)

with stat_col1:
    st.subheader("📊 Object Count")
    count_placeholder = st.empty()

with stat_col2:
    st.subheader("🚨 Active Alerts")
    alert_placeholder = st.empty()

# Render latest counts & alerts
with lock:
    current_counts = dict(shared_state["counts"])
    current_alerts = list(shared_state["alerts"])

if current_counts:
    count_md = "\n".join([f"- **{k}**: {v}" for k, v in sorted(current_counts.items())])
    count_placeholder.markdown(count_md)
else:
    count_placeholder.info("No objects detected yet.")

if current_alerts:
    for alert in current_alerts:
        alert_placeholder.warning(alert)
else:
    alert_placeholder.success("No alerts triggered.")

# ── Saved frames gallery ──────────────────────────────────────────────────────
st.divider()
st.subheader("🖼️ Saved Frames Gallery")

saved_files = sorted(
    [f for f in os.listdir(SAVE_DIR) if f.endswith(".jpg")],
    reverse=True,
)

if saved_files:
    cols = st.columns(min(len(saved_files), 4))
    for i, fname in enumerate(saved_files[:8]):   # show latest 8
        img_path = os.path.join(SAVE_DIR, fname)
        cols[i % 4].image(img_path, caption=fname, use_column_width=True)
else:
    st.info("No frames saved yet. Click '📷 Capture current frame' during detection.")