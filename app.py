import streamlit as st
import pandas as pd
import numpy as np
import time
import cv2
import os
import joblib
from datetime import datetime
from PIL import Image
import sys

# Ensure Project_Amine is in path for imports
if os.path.exists("Project_Amine"):
    sys.path.append(os.path.abspath("Project_Amine"))
    sys.path.append(os.getcwd())
elif os.path.exists("Project-Amine"):
    sys.path.append(os.path.abspath("Project-Amine"))
    sys.path.append(os.getcwd())

# Import logic using relative paths
try:
    from iot.simulator import IoTSimulator
    from camera.video_simulator import VideoSimulator
    from camera.detector import CameraDetector
    from utils.config import IOT_CONFIG, CAMERA_CONFIG
    from models.classifier import IntrusionClassifier
    from anomalies.ensemble import AnomalyEnsemble
except ImportError:
    try:
        from Project_Amine.iot.simulator import IoTSimulator
        from Project_Amine.camera.video_simulator import VideoSimulator
        from Project_Amine.camera.detector import CameraDetector
        from Project_Amine.utils.config import IOT_CONFIG, CAMERA_CONFIG
        from Project_Amine.models.classifier import IntrusionClassifier
        from Project_Amine.anomalies.ensemble import AnomalyEnsemble
    except ImportError as e:
        st.error(f"Erreur d'importation : {e}")
        st.stop()

# --- Page Configuration ---
st.set_page_config(
    page_title="GuardianAI | Enterprise Command",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Enhanced UI Styling ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    :root {
        --primary: #00f2fe;
        --secondary: #4facfe;
        --bg-dark: #06090f;
        --card-bg: rgba(13, 17, 23, 0.8);
        --border: rgba(255, 255, 255, 0.08);
        --accent-red: #ff4d4d;
        --accent-orange: #ffa500;
        --accent-green: #00ff88;
        --glow: 0 0 20px rgba(79, 172, 254, 0.25);
    }

    /* Global Overrides */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #1c2533, #06090f) !important;
        color: #f0f6fc !important;
        font-family: 'Space Grotesk', sans-serif;
    }

    /* Professional Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #010409 !important;
        border-right: 1px solid var(--border) !important;
    }

    /* Fix visibility of text and labels in sidebar */
    [data-testid="stSidebar"] .stMarkdown p, 
    [data-testid="stSidebar"] .stMarkdown span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stCheckbox p,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        color: #ffffff !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        opacity: 1 !important;
    }

    /* Contrast for Expanders */
    [data-testid="stSidebar"] [data-testid="stExpander"] {
        background: transparent !important;
        border: 1px solid var(--border) !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stExpander"] summary {
        background-color: transparent !important;
        color: #ffffff !important;
    }

    [data-testid="stSidebar"] [data-testid="stExpander"] summary p {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
    }

    /* Slider values and text */
    [data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stMarkdownContainer"] p {
        color: var(--secondary) !important;
        font-weight: 700 !important;
    }

    /* Force all text inside sidebar to be white/readable */
    [data-testid="stSidebar"] * {
        color: #ffffff;
    }

    /* Exceptions for specific accents */
    [data-testid="stSidebar"] .stSlider * {
        color: inherit;
    }

    /* Metric Styling for high visibility */
    [data-testid="stMetricValue"] {
        font-size: 2.4rem !important;
        font-weight: 800 !important;
        color: #00f2fe !important; /* Bright Cyan */
        text-shadow: 0 0 15px rgba(0, 242, 254, 0.5);
    }
    
    [data-testid="stMetricLabel"] p {
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        letter-spacing: 1.5px !important;
        text-transform: uppercase !important;
        margin-bottom: 5px !important;
    }

    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 16px !important;
        padding: 20px !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
        transition: transform 0.3s ease !important;
    }

    [data-testid="stMetric"]:hover {
        transform: translateY(-5px) !important;
        border-color: var(--secondary) !important;
        background: rgba(79, 172, 254, 0.08) !important;
    }

    .section-title {
        font-size: 0.8rem;
        font-weight: 700;
        color: var(--secondary);
        letter-spacing: 2.5px;
        text-transform: uppercase;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* Model Badge */
    .model-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 8px;
        background: rgba(79, 172, 254, 0.15);
        color: var(--secondary);
        border: 1px solid rgba(79, 172, 254, 0.3);
    }
    
    .best-badge {
        background: rgba(0, 255, 136, 0.15);
        color: var(--accent-green);
        border: 1px solid rgba(0, 255, 136, 0.3);
    }

    /* Metric Grid */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
        gap: 10px;
    }

    .mini-metric {
        padding: 10px;
        background: rgba(255,255,255,0.03);
        border-radius: 8px;
        text-align: center;
    }

    /* Glass Card */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 15px;
    }
</style>
""", unsafe_allow_html=True)

# --- Model Loading ---
@st.cache_resource
def load_ai_models():
    classifier = IntrusionClassifier()
    classifier.load_models()
    anomaly_ensemble = AnomalyEnsemble()
    anomaly_ensemble.load_all()
    # Initialize Camera Detector for person detection & tracking
    cam_detector = CameraDetector()
    return classifier, anomaly_ensemble, cam_detector

classifier, anomaly_ensemble, cam_detector = load_ai_models()

# --- Sidebar ---
with st.sidebar:
    st.markdown("<h2 style='color:#4facfe;'>COMMAND CENTER</h2>", unsafe_allow_html=True)
    
    with st.expander("📡 IOT TELEMETRY", expanded=True):
        manual_inputs = {}
        units = {
            "vibration": "v",
            "audio": "dB",
            "temperature": "°C",
            "co2": "ppm"
        }
        for sensor, cfg in IOT_CONFIG["sensors"].items():
            if sensor == "pir_motion":
                manual_inputs[sensor] = st.checkbox(f"PIR Discovery", value=False)
            else:
                normal_range = cfg.get("normal_range", [cfg["min"], cfg["max"]])
                normal_mid = (normal_range[0] + normal_range[1]) / 2
                unit = units.get(sensor, "")
                manual_inputs[sensor] = st.slider(
                    f"{sensor.upper()} ({unit})",
                    float(cfg["min"]), float(cfg["max"]), float(normal_mid)
                )

    with st.expander("🧠 AI PROTOCOLS", expanded=True):
        include_intruder = st.toggle("Human Signature Simulation", value=False)
        duration = st.select_slider("Burst Depth", options=[60, 120, 240, 480], value=60)
        record_video = st.toggle("Protocol: Auto-Archive", value=True)

    launch_btn = st.button("EXECUTE NEURAL SCAN", width='stretch', type="primary")

# --- Header ---
st.markdown("""
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:30px; background:rgba(255,255,255,0.02); padding:25px; border-radius:15px; border:1px solid rgba(255,255,255,0.05);">
    <div>
        <h1 style="margin:0; font-size:2.8rem;">GuardianAI <span style="color:rgba(255,255,255,0.2)">v2.0</span></h1>
        <p style="margin:0; color:rgba(255,255,255,0.4);">Advanced Neural Surveillance & Intrusion Intelligence</p>
    </div>
    <div style="text-align:right;">
        <span style="background:#00ff8822; color:#00ff88; padding:5px 15px; border-radius:10px; font-weight:700; font-size:0.8rem; border:1px solid #00ff8844;">SYSTEM READY</span>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Main Layout ---
col_main, col_side = st.columns([2, 1], gap="large")

with col_main:
    st.markdown('<div class="section-title">Visual Intelligence Stream</div>', unsafe_allow_html=True)
    st.markdown('<div class="glass-card" style="padding:10px;">', unsafe_allow_html=True)
    video_placeholder = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Model Decision Matrix</div>', unsafe_allow_html=True)
    model_comparison_placeholder = st.empty()

with col_side:
    st.markdown('<div class="section-title">Threat Assessment</div>', unsafe_allow_html=True)
    status_placeholder = st.empty()
    
    st.markdown('<div class="section-title">Model Attribution</div>', unsafe_allow_html=True)
    attribution_placeholder = st.empty()

    st.markdown('<div class="section-title">Event Timeline</div>', unsafe_allow_html=True)
    terminal_placeholder = st.empty()

# --- Logic ---
if 'terminal_logs' not in st.session_state:
    st.session_state.terminal_logs = []

def add_smart_log(msg, level="info"):
    ts = datetime.now().strftime("%H:%M:%S")
    colors = {"info": "#4facfe", "warning": "#ffa500", "error": "#ff4d4d", "success": "#00ff88"}
    line = f'<div style="color:{colors[level]}; font-family:monospace; margin-bottom:5px;">[{ts}] {msg}</div>'
    st.session_state.terminal_logs.append(line)
    st.session_state.terminal_logs = st.session_state.terminal_logs[-15:]

if launch_btn:
    st.session_state.terminal_logs = []
    add_smart_log("Simulation initiated...", "info")
    
    # Reset camera detector state for new run
    if cam_detector.tracker:
        cam_detector.tracker.reset()
    cam_detector.recording = False
    cam_detector.recording_buffer = []

    prog = st.progress(0)
    video_sim = VideoSimulator()

    for i in range(duration):
        # Get frame with optional intruder
        raw_frame, sim_detections = video_sim.get_frame(include_intruder=include_intruder)
        
        # Process frame with YOLO + Tracking
        # Use our CameraDetector which handles YOLO + DeepSort-style tracking
        cam_result = cam_detector.process_frame(raw_frame, sim_detections)
        
        # Annotate frame for display
        display_frame = cam_detector.draw_detections(raw_frame, cam_result)
        
        # IOT Data Processing
        input_data = [
            manual_inputs["vibration"],
            manual_inputs["audio"],
            manual_inputs["temperature"],
            manual_inputs["co2"],
            1.0 if manual_inputs["pir_motion"] else 0.0
        ]
        
        ensemble_res = classifier.predict_ensemble(input_data)
        anom_res = anomaly_ensemble.predict_point(np.array(input_data))
        
        # Determine anomaly description
        anomaly_msg = ""
        anomaly_details = []
        is_anomaly_detected = anom_res.get("ensemble", {}).get("is_anomaly", False)
        if is_anomaly_detected:
            anom_types = []
            if anom_res.get("isolation_forest", {}).get("is_anomaly"): anom_types.append("Isolation Forest")
            if anom_res.get("autoencoder", {}).get("is_anomaly"): anom_types.append("Autoencoder")
            anomaly_msg = " | Source: " + ", ".join(anom_types) if anom_types else ""
            
            # Identify which sensor might be causing the anomaly
            sensor_names = ["vibration", "audio", "temperature", "co2", "pir_motion"]
            for sensor, value in zip(sensor_names, input_data):
                cfg = IOT_CONFIG["sensors"].get(sensor, {})
                normal_range = cfg.get("normal_range", [cfg.get("min", 0), cfg.get("max", 100)])
                if value < normal_range[0] or value > normal_range[1]:
                    unit = units.get(sensor, "")
                    anomaly_details.append(f"{sensor.upper()} anormal ({value}{unit})")
        
        detail_text = " | " + ", ".join(anomaly_details) if anomaly_details else ""
        
        # Smart Decision Fusion
        is_human = cam_result["is_intrusion"]
        is_iot_intrusion = any(m["prediction"] == 1 for m in ensemble_res.get("models", {}).values())
        
        # Comprehensive status logic
        status_parts = []
        if is_human:
            status_parts.append("INTRUSION HUMAINE 👤")
        if is_iot_intrusion:
            status_parts.append("INTRUSION CAPTEURS 📡")
        
        is_intrusion = len(status_parts) > 0
        
        if is_intrusion and is_anomaly_detected:
            f_s = " + ".join(status_parts) + " (AVEC ANOMALIE)"
        elif is_intrusion:
            f_s = " + ".join(status_parts)
        elif is_anomaly_detected:
            f_s = "ANOMALIE SYSTÈME ⚠️"
        else:
            f_s = "MAISON SÉCURISÉE 🟢"
        
        clr = "#00ff88" if "SÉCURISÉE" in f_s else "#ff4d4d" if is_intrusion else "#ffa500"

        # Update Visuals
        video_placeholder.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), width='stretch')
        
        # Update Status
        diag_html = ""
        if is_human:
            p_count = cam_result["person_count"]
            diag_html += f'<div style="font-size:0.8rem; color:#ff4d4d; margin-top:10px; font-weight:600;">👤 DETECTION: {p_count} humain(s) (YOLOv8 + DeepSort)</div>'
        if is_anomaly_detected:
            diag_html += f'<div style="font-size:0.8rem; color:#ffa500; margin-top:5px; font-weight:600;">⚠️ DIAGNOSTIC: {anomaly_msg}{detail_text}</div>'
        
        # Notifications
        if (is_intrusion or is_anomaly_detected) and i % 20 == 0:
            st.toast(f"🚨 {f_s}", icon="🛡️")
            add_smart_log(f"ALERT DISPATCHED: {f_s}", "error")

        status_placeholder.markdown(f"""
        <div class="glass-card" style="text-align:center; border-color:{clr}44;">
            <div style="font-size:0.7rem; color:rgba(255,255,255,0.4); margin-bottom:10px;">THREAT LEVEL</div>
            <div style="font-size:1.8rem; font-weight:900; color:{clr};">{f_s}</div>
            {diag_html}
            <div style="font-size:0.8rem; margin-top:10px;">Confidence: {ensemble_res["avg_confidence"]*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        # Update Attribution
        best_model = ensemble_res["best_model"].upper()
        attribution_placeholder.markdown(f"""
        <div class="glass-card" style="background:rgba(0,255,136,0.05); border-color:#00ff8833;">
            <div style="font-size:0.7rem; color:rgba(255,255,255,0.4); margin-bottom:5px;">ACTIVE EXPERT MODEL</div>
            <div style="font-size:1.1rem; font-weight:700; color:#00ff88;">{best_model}</div>
            <div style="font-size:0.8rem; color:rgba(255,255,255,0.6); margin-top:5px;">Leading with highest local confidence.</div>
        </div>
        """, unsafe_allow_html=True)

        # Update Terminal
        terminal_placeholder.markdown(f"""
        <div style="height:200px; overflow-y:auto; background:rgba(0,0,0,0.3); padding:10px; border-radius:5px; border:1px solid rgba(255,255,255,0.05);">
            {''.join(st.session_state.terminal_logs[::-1])}
        </div>
        """, unsafe_allow_html=True)

        prog.progress((i+1)/duration)
        time.sleep(0.01)

    # Save final recording if any intrusion happened
    if cam_detector.recording:
        # The CameraDetector._stop_recording() handles drawing detections on all frames in the video
        cam_detector._stop_recording()
        # Find the latest recording
        recordings_dir = "outputs/recordings"
        if os.path.exists(recordings_dir):
            files = [os.path.join(recordings_dir, f) for f in os.listdir(recordings_dir) if f.endswith(".avi")]
            if files:
                latest_recording = max(files, key=os.path.getctime)
                add_smart_log(f"Protocol: Intrusion archive generated at {latest_recording}", "success")
                st.info(f"Dernier enregistrement (avec tracking) : {latest_recording}")

    # --- Neural Performance Metrics ---
    st.markdown('<div class="section-title">Model Decision Matrix</div>', unsafe_allow_html=True)
    m_c1, m_c2, m_c3 = st.columns(3)
    models_res = ensemble_res.get("models", {})
    rf_conf = models_res.get("random_forest", {}).get("confidence", 0)
    svm_conf = models_res.get("svm", {}).get("confidence", 0)
    xgb_conf = models_res.get("xgboost", {}).get("confidence", 0)
    
    with m_c1: st.metric("Random Forest", f"{rf_conf*100:.1f}%")
    with m_c2: st.metric("SVM Predictor", f"{svm_conf*100:.1f}%")
    with m_c3: st.metric("XGBoost Engine", f"{xgb_conf*100:.1f}%")
