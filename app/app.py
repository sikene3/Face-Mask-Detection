import os
import time
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO
from PIL import Image
from datetime import datetime
import plotly.graph_objects as go

st.set_page_config(page_title="VisionGuard AI", page_icon="🛡️", layout="wide")

CLASS_NAMES = ["with_mask", "without_mask", "mask_weared_incorrect"]
CLASS_COLORS = {
    "with_mask": (0, 255, 0),
    "without_mask": (0, 0, 255),
    "mask_weared_incorrect": (0, 255, 255),
}
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');

    html, body, [class*="css"] {
        font-family: 'Orbitron', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #0d1117 50%, #0a0a0a 100%);
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #0a0a0a 100%);
        border-right: 2px solid #00ffcc;
        box-shadow: 0 0 20px rgba(0, 255, 204, 0.15);
    }

    section[data-testid="stSidebar"] * {
        color: #00ffcc !important;
    }

    h1, h2, h3 {
        color: #00ffcc !important;
        text-shadow: 0 0 10px rgba(0, 255, 204, 0.5);
        letter-spacing: 2px;
    }

    .stButton > button {
        background: transparent;
        color: #00ffcc;
        border: 2px solid #00ffcc;
        border-radius: 4px;
        font-family: 'Orbitron', sans-serif;
        text-transform: uppercase;
        letter-spacing: 2px;
        transition: all 0.3s ease;
        box-shadow: 0 0 10px rgba(0, 255, 204, 0.2);
    }

    .stButton > button:hover {
        background: #00ffcc;
        color: #0a0a0a;
        box-shadow: 0 0 25px rgba(0, 255, 204, 0.6);
    }

    .stSlider > div > div > div {
        background: #00ffcc !important;
    }

    .stFileUploader {
        border: 2px dashed #00ffcc !important;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 0 15px rgba(0, 255, 204, 0.1);
    }

    .stMetric {
        background: rgba(0, 255, 204, 0.05);
        border: 1px solid #00ffcc;
        border-radius: 8px;
        padding: 10px;
    }

    .stMetric label {
        color: #00ffcc !important;
    }

    .stDataFrame {
        border: 1px solid #00ffcc !important;
        border-radius: 8px;
    }

    .stDataFrame th {
        background: #00ffcc !important;
        color: #0a0a0a !important;
    }

    .stDataFrame td {
        color: #00ffcc !important;
        background: rgba(0, 255, 204, 0.03) !important;
    }

    .stAlert {
        border: 1px solid #00ffcc !important;
        background: rgba(0, 255, 204, 0.05) !important;
    }

    div[data-testid="stNotification"] {
        background: rgba(255, 0, 68, 0.15) !important;
        border: 1px solid #ff0044 !important;
    }

    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        st.error(f"Model not found at: {path}")
        st.stop()
    return YOLO(path)


model = load_model(MODEL_PATH)


def run_inference(image, conf_threshold):
    results = model(image, conf=conf_threshold, verbose=False)
    return results[0]


def draw_boxes(image_bgr, result):
    boxes = result.boxes
    if boxes is None:
        return image_bgr
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = CLASS_NAMES[cls_id]
        color = CLASS_COLORS.get(label, (255, 255, 255))
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_y = y1 - 4 if y1 - th - 4 > 0 else y1 + th + 4
        cv2.rectangle(
            image_bgr,
            (x1, text_y - th - 2),
            (x1 + tw + 2, text_y + 2),
            color,
            -1,
        )
        cv2.putText(
            image_bgr,
            text,
            (x1 + 1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return image_bgr


def get_class_counts(result):
    counts = {name: 0 for name in CLASS_NAMES}
    if result.boxes is None:
        return counts
    for box in result.boxes:
        cls_id = int(box.cls[0])
        counts[CLASS_NAMES[cls_id]] += 1
    return counts


def compliance_status(counts):
    total = sum(counts.values())
    if total == 0:
        return "NO FACES DETECTED", "⚪"
    compliant = counts["with_mask"]
    ratio = compliant / total if total > 0 else 0
    if ratio >= 0.8:
        return "COMPLIANT", "🟢"
    elif ratio >= 0.5:
        return "PARTIALLY COMPLIANT", "🟡"
    else:
        return "NON-COMPLIANT", "🔴"


if "webcam_running" not in st.session_state:
    st.session_state.webcam_running = False
if "cumulative_counts" not in st.session_state:
    st.session_state.cumulative_counts = {name: 0 for name in CLASS_NAMES}
if "audit_log" not in st.session_state:
    st.session_state.audit_log = []
if "frame_count" not in st.session_state:
    st.session_state.frame_count = 0
if "snapshot_requested" not in st.session_state:
    st.session_state.snapshot_requested = False


with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding:20px 0;">
            <h1 style="font-size:2.2rem; margin:0;">🛡️ VISIONGUARD</h1>
            <p style="color:#00ffcc; opacity:0.7; letter-spacing:3px; margin:0;">AI</p>
        </div>
        <hr style="border-color:#00ffcc; opacity:0.3;">
        """,
        unsafe_allow_html=True,
    )

    prev_mode = st.session_state.get("detection_mode", "📷 Image Upload")
    mode = st.radio(
        "DETECTION MODE",
        ["📷 Image Upload", "🎥 Live Webcam"],
        index=0 if "Image" in prev_mode else 1,
        label_visibility="visible",
        key="detection_mode",
    )

    if prev_mode != mode and "Live Webcam" not in mode:
        st.session_state.webcam_running = False

    st.markdown("<br>", unsafe_allow_html=True)

    conf_threshold = st.slider(
        "CONFIDENCE THRESHOLD",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Minimum confidence score for detections to be displayed.",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🗑 RESET ANALYTICS", use_container_width=True):
        st.session_state.cumulative_counts = {name: 0 for name in CLASS_NAMES}
        st.session_state.audit_log = []
        st.session_state.frame_count = 0
        st.rerun()

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="text-align:center; opacity:0.5; font-size:0.7rem;">
            <p>YOLOv8 Nano · Face Mask Detection</p>
            <p style="color:#00ffcc;">v2.0.0 Enterprise</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    f"""
    <div style="text-align:center; padding:10px 0 30px 0;">
        <h1 style="font-size:3rem; margin:0;">
            {'📷 IMAGE ANALYSIS' if 'Image' in mode else '🎥 LIVE MONITORING'}
        </h1>
        <p style="color:#00ffcc; opacity:0.6; letter-spacing:4px;">REAL-TIME FACE MASK COMPLIANCE</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if "Image Upload" in mode:
    uploaded_file = st.file_uploader(
        "DROP YOUR IMAGE HERE",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                "<p style='color:#00ffcc; text-align:center;'>ORIGINAL</p>",
                unsafe_allow_html=True,
            )
            original_img = Image.open(uploaded_file).convert("RGB")
            st.image(original_img, use_container_width=True)

        with col2:
            st.markdown(
                "<p style='color:#00ffcc; text-align:center;'>DETECTIONS</p>",
                unsafe_allow_html=True,
            )
            img_bgr = cv2.cvtColor(np.array(original_img), cv2.COLOR_RGB2BGR)
            result = run_inference(img_bgr, conf_threshold)
            annotated_bgr = draw_boxes(img_bgr, result)
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, use_container_width=True)

        counts = get_class_counts(result)
        status, icon = compliance_status(counts)

        st.markdown("<br>", unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("✅ WITH MASK", counts["with_mask"])
        with m2:
            st.metric("❌ WITHOUT MASK", counts["without_mask"])
        with m3:
            st.metric("⚠️ INCORRECT", counts["mask_weared_incorrect"])
        with m4:
            st.metric("COMPLIANCE", f"{icon} {status}")

        st.markdown("<br>", unsafe_allow_html=True)

        df = pd.DataFrame(
            [
                {
                    "Class": name,
                    "Count": counts[name],
                    "Status": (
                        "Compliant"
                        if name == "with_mask"
                        else "Non-Compliant"
                    ),
                }
                for name in CLASS_NAMES
            ]
        )
        st.dataframe(df, use_container_width=True, hide_index=True)

elif "Live Webcam" in mode:
    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        label = "⏹ STOP STREAM" if st.session_state.webcam_running else "▶ START STREAM"
        if st.button(label, use_container_width=True):
            st.session_state.webcam_running = not st.session_state.webcam_running
            st.rerun()

    if st.session_state.webcam_running:
        frame_placeholder = st.empty()
        alert_placeholder = st.empty()
        metrics_placeholder = st.empty()
        audit_placeholder = st.empty()
        chart_placeholder = st.empty()

        snap_col1, snap_col2 = st.columns([1, 3])
        with snap_col1:
            if st.button("📸 CAPTURE VIOLATION", use_container_width=True, key="snap_btn"):
                st.session_state.snapshot_requested = True
                st.rerun()

        snapshots_dir = os.path.join(BASE_DIR, "outputs", "snapshots")
        os.makedirs(snapshots_dir, exist_ok=True)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not open webcam. Check camera permissions.")
            st.session_state.webcam_running = False
            st.stop()

        prev_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to read frame from webcam.")
                break

            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time

            results = model(frame, conf=conf_threshold, verbose=False)
            annotated = results[0].plot()

            cv2.putText(
                annotated,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 204),
                2,
                cv2.LINE_AA,
            )

            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(annotated_rgb, channels="RGB", use_container_width=True)

            if st.session_state.snapshot_requested:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(snapshots_dir, f"violation_{ts}.png")
                cv2.imwrite(filename, annotated)
                st.session_state.snapshot_requested = False

            counts = get_class_counts(results[0])
            for name in CLASS_NAMES:
                st.session_state.cumulative_counts[name] += counts[name]
            st.session_state.frame_count += 1

            has_violation = counts["without_mask"] > 0
            if has_violation and results[0].boxes is not None:
                ts = datetime.now().strftime("%H:%M:%S")
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    if CLASS_NAMES[cls_id] == "without_mask":
                        conf = float(box.conf[0])
                        st.session_state.audit_log.append({
                            "Timestamp": ts,
                            "Status": "🚨 VIOLATION",
                            "Confidence": f"{conf:.2f}",
                        })
                if len(st.session_state.audit_log) > 50:
                    st.session_state.audit_log = st.session_state.audit_log[-50:]

            with alert_placeholder.container():
                if has_violation:
                    st.error("🚨 VIOLATION DETECTED! Unmasked individual(s) found.")
                else:
                    st.success("✅ ALL CLEAR — No violations detected.")

            status, icon = compliance_status(counts)
            with metrics_placeholder.container():
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("✅ WITH MASK", counts["with_mask"])
                with m2:
                    st.metric("❌ WITHOUT MASK", counts["without_mask"])
                with m3:
                    st.metric("⚠️ INCORRECT", counts["mask_weared_incorrect"])
                with m4:
                    st.metric("COMPLIANCE", f"{icon} {status}")

            with audit_placeholder.container():
                st.markdown(
                    "<p style='color:#00ffcc;'>🔍 SECURITY AUDIT LOG (Last 5)</p>",
                    unsafe_allow_html=True,
                )
                if st.session_state.audit_log:
                    df_audit = pd.DataFrame(st.session_state.audit_log[-5:])
                    st.dataframe(df_audit, use_container_width=True, hide_index=True)
                else:
                    st.info("No violations recorded yet.")

            if st.session_state.frame_count % 5 == 0:
                with chart_placeholder.container():
                    st.markdown(
                        "<p style='color:#00ffcc;'>📊 LIVE COMPLIANCE ANALYTICS</p>",
                        unsafe_allow_html=True,
                    )
                    cum = st.session_state.cumulative_counts
                    total = sum(cum.values())
                    if total > 0:
                        fig = go.Figure(data=[go.Pie(
                            labels=list(cum.keys()),
                            values=list(cum.values()),
                            hole=0.5,
                            marker=dict(colors=["#00ff00", "#ff0044", "#ffcc00"]),
                            textinfo="label+percent",
                            textfont=dict(color="#00ffcc", family="Orbitron"),
                        )])
                        fig.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#00ffcc", family="Orbitron"),
                            margin=dict(l=20, r=20, t=30, b=20),
                            height=350,
                            showlegend=False,
                        )
                        st.plotly_chart(
                            fig,
                            use_container_width=True,
                            key=f"donut_{st.session_state.frame_count}",
                        )
                    else:
                        st.info("Waiting for detections...")

        cap.release()
        st.session_state.webcam_running = False
        st.rerun()
