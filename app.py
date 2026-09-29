import streamlit as st
import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
import librosa
from extract_features import extract_features

st.set_page_config(page_title="Disaster Sound AI", page_icon="📡", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .navbar { background-color: #1e293b; padding: 15px 30px; border-radius: 8px; margin-bottom: 25px; display: flex; justify-content: space-between; align-items: center; color: white; }
    .nav-brand { font-size: 22px; font-weight: bold; color: #38bdf8; }
    .nav-status { background-color: #065f46; color: #34d399; padding: 4px 12px; border-radius: 12px; font-size: 14px; font-weight: 600; }
    .footer { margin-top: 80px; padding: 20px; background-color: #f8fafc; border-top: 1px solid #e2e8f0; border-radius: 8px; text-align: center; font-size: 14px; color: #64748b; }
    </style>
""", unsafe_allow_html=True)

# --- NAVBAR ---
st.markdown('<div class="navbar"><div class="nav-brand">🛰️ Disaster Sound AI</div><div><span style="margin-right:20px;color:#94a3b8;">v2.0.0 (Multi-Class)</span><span class="nav-status">● SYSTEM LIVE</span></div></div>', unsafe_allow_html=True)

MODEL_PATH = "disaster_model.joblib"
temp_filename = "temp_live_stream.wav"

if not os.path.exists(MODEL_PATH):
    st.error(f"❌ System Fault: Pre-trained brain asset file '{MODEL_PATH}' missing.")
    st.stop()

# --- SIDEBAR WAVEFORM GENERATOR ---
with st.sidebar:
    st.markdown("## 📡 Network Node Status")
    st.info("🟢 Edge Node: **Active**  \n📶 Signal: **Strong**  \n🔋 Battery: **98%**")
    st.markdown("---")
    st.markdown("### 📊 Live Telemetry Waveform")
    
    if os.path.exists(temp_filename):
        try:
            y_wave, sr_wave = librosa.load(temp_filename, sr=22050)
            time_axis = np.linspace(0, len(y_wave) / sr_wave, num=len(y_wave))
            fig, ax = plt.subplots(figsize=(4, 2.5))
            ax.plot(time_axis, y_wave, color='#38bdf8', alpha=0.8, linewidth=1)
            ax.set_title("Signal Amplitude vs Time", fontsize=10, color='#1e293b', weight='bold')
            ax.set_xlabel("Time (seconds)", fontsize=8)
            ax.grid(True, linestyle='--', alpha=0.5)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        except:
            st.caption("Initializing matrix graphics...")
    else:
        st.markdown("<center style='color:#64748b; font-style:italic; padding:20px 0;'>No active incoming telemetry stream.</center>", unsafe_allow_html=True)
    st.markdown("---")
    st.caption("Acoustic Sentinel Core v2.0.0")

# --- NAVIGATION TABS ---
tab_dashboard, tab_tech, tab_about = st.tabs(["🖥️ Live Telemetry Dashboard", "⚙️ Model & Core Architecture", "ℹ️ Project Information"])

with tab_dashboard:
    st.markdown("### 🛰️ Real-Time 5-Tier Disaster Processing Dashboard")
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("#### 📥 1. Environmental Audio Feed")
        with st.container(border=True):
            uploaded_file = st.file_uploader("Drop a .wav telemetry audio stream file here", type=["wav"], label_visibility="collapsed")
            if uploaded_file is not None:
                with open(temp_filename, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.markdown("##### 🎵 Signal Audio Player")
                st.audio(uploaded_file, format="audio/wav")
                if 'file_uploaded' not in st.session_state or st.session_state.file_uploaded != uploaded_file.name:
                    st.session_state.file_uploaded = uploaded_file.name
                    st.rerun()

    with col2:
        st.markdown("#### 🔮 2. Live Translation Analytics")
        if uploaded_file is None:
            with st.container(border=True):
                st.markdown("<br><center>🤖 <i>Waiting for active audio telemetry stream...</i></center><br>", unsafe_allow_html=True)
        
        if uploaded_file is not None:
            if st.button("⚡ Execute Frequential Translation", use_container_width=True, type="primary"):
                with st.spinner("Processing structural telemetry matrices..."):
                    fingerprint = extract_features(temp_filename)
                    if fingerprint is None:
                        st.error("❌ Signal decoding failed.")
                    else:
                        loaded_model = joblib.load(MODEL_PATH)
                        prediction = loaded_model.predict(fingerprint.reshape(1, -1))[0]
                        probabilities = loaded_model.predict_proba(fingerprint.reshape(1, -1))[0]
                        class_labels = loaded_model.classes_
                        
                        winning_idx = np.where(class_labels == prediction)[0][0]
                        confidence_score = probabilities[winning_idx] * 100
                        
                        with st.container(border=True):
                            st.markdown("##### 🚨 Decoded Output Alert Message")
                            
                            # Distinct visual card styles tailored specifically per class prediction
                            if "Safe" in prediction:
                                color_bg, color_border, status_title = "#e6f4ea", "#137333", "🟢 NORMAL AMBIENT STATUS"
                            elif "Fracture" in prediction:
                                color_bg, color_border, status_title = "#feeedb", "#b06000", "⚠️ INFRASTRUCTURE WARNING"
                            else:
                                color_bg, color_border, status_title = "#fce8e6", "#c5221f", "🛑 CRITICAL EMERGENCY EVACUATION"
                            
                            st.markdown(f"""
                                <div style='background-color:{color_bg}; padding:15px; border-left: 6px solid {color_border}; border-radius:4px;'>
                                <h3 style='color:{color_border}; margin:0;'>{status_title}</h3>
                                <p style='color:{color_border}; font-size:16px; margin-top:5px;'><b>[ Broadcast: {prediction} ]</b></p>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown(f"<br>**AI Classification Certainty:** `{confidence_score:.2f}%`", unsafe_allow_html=True)
                            st.progress(int(confidence_score))
                            st.markdown("---")
                            
                            # Dynamic listing of all 5 class probability vectors
                            st.markdown("##### 📊 Full Frequential Vector Probabilities")
                            for label, prob in zip(class_labels, probabilities):
                                st.write(f"- **{label}:** `{prob*100:.1f}%`")

with tab_tech:
    st.markdown("### ⚙️ Deep-Tech Technical Specifications")
    st.write("Current configuration running an advanced **Multi-Class Abstraction Framework** mapping 5 separate acoustic anomalies.")

with tab_about:
    st.markdown("### ℹ️ About Acoustic Sentinel Multi-Class v2.0")
    st.write("Optimized dashboard handling nuanced disaster variances: differentiating normal background rain from flash floods, structural concrete stress events, wildfires, and active landslides.")

st.markdown('<div class="footer">🛰️ <b>Acoustic Sentinel v2.0 Multi-Class Project</b> | Built using Python, Librosa, Scikit-Learn, and Streamlit</div>', unsafe_allow_html=True)
