import streamlit as st
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import scipy.io.wavfile as wf
from io import BytesIO
from utils.wav_utils import _read_wav, record_audio
from models.energy_vad import run_energy_vad
from models.silero_vad import run_silero_vad
from methods.spectral_entropy_vad import entropy_vad_segments
from methods.zcr_vad import zcr_vad_segments

st.set_page_config(page_title="Voice Activity Detection", layout="wide")
st.title("\U0001F399️ Voice Activity Detection (VAD) App")

mode = st.radio("Select Input Mode", ["Upload WAV File", "Real-time Microphone"])

if mode == "Upload WAV File":
    uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])
    if uploaded_file:
        rate, data = _read_wav(uploaded_file)
        st.audio(uploaded_file, format='audio/wav', sample_rate=rate)
elif mode == "Real-time Microphone":
    duration = st.slider("Recording Duration (s)", 1, 10, 5)
    if st.button("\U0001F3A4 Record Audio"):
        rate, data = record_audio(duration=duration)
        
        # Convert the data into byte format and play it back using Streamlit
        audio_bytes = BytesIO()
        wf.write(audio_bytes, rate, (data * 32767).astype(np.int16))
        audio_bytes.seek(0)
        st.audio(audio_bytes, format='audio/wav', sample_rate=rate)

# Sidebar parameters
st.sidebar.header("VAD Settings")
methods = ["Energy-Based", "Silero", "Spectral Entropy", "ZCR"]
method = st.sidebar.selectbox("VAD Method", methods)
SAMPLE_WINDOW = st.sidebar.slider("Window Size (s)", 0.01, 0.1, 0.02, 0.005)
SAMPLE_OVERLAP = st.sidebar.slider("Overlap (s)", 0.005, 0.05, 0.01, 0.005)
THRESHOLD = st.sidebar.slider("Voice Energy Threshold", 0.1, 0.9, 0.6, 0.05)

if 'data' in locals() and data is not None:
    if method == "Energy-Based":
        run_energy_vad(data, rate, SAMPLE_WINDOW, SAMPLE_OVERLAP, THRESHOLD)
    elif method == "Silero":
        run_silero_vad(data, rate)
    elif method == "Spectral Entropy":
        segments = entropy_vad_segments(data, rate, window_duration=SAMPLE_WINDOW, threshold=THRESHOLD)
        # Plot and display segments for Spectral Entropy VAD
        fig, ax = plt.subplots(figsize=(12, 3))
        time = np.arange(len(data)) / rate
        ax.plot(time, data, alpha=0.5)
        for seg in segments:
            ax.axvspan(seg[0], seg[1], color='green', alpha=0.3)
        st.pyplot(fig)

        st.subheader("\U0001F4CC Detected Segments")
        for i, seg in enumerate(segments):
            st.markdown(f"**Segment {i+1}: {seg[0]:.2f}s - {seg[1]:.2f}s**")
        
    elif method == "ZCR":
        segments = zcr_vad_segments(data, rate, window_duration=SAMPLE_WINDOW, threshold=THRESHOLD)
        # Plot and display segments for ZCR VAD
        fig, ax = plt.subplots(figsize=(12, 3))
        time = np.arange(len(data)) / rate
        ax.plot(time, data, alpha=0.5)
        for seg in segments:
            ax.axvspan(seg[0], seg[1], color='green', alpha=0.3)
        st.pyplot(fig)

        st.subheader("\U0001F4CC Detected Segments")
        for i, seg in enumerate(segments):
            st.markdown(f"**Segment {i+1}: {seg[0]:.2f}s - {seg[1]:.2f}s**")
