import streamlit as st
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import scipy.io.wavfile as wf
from utils.wav_utils import _read_wav, record_audio
from models.energy_vad import run_energy_vad
from models.webrtc_vad import apply_webrtc_vad
from models.silero_vad import run_silero_vad
from models.wav2vec_vad import run_wav2vec_vad

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
        st.audio(data, format='audio/wav', sample_rate=rate)

# Sidebar parameters
st.sidebar.header("VAD Settings")
methods = ["Energy-Based", "WebRTC", "Silero", "Wav2Vec2"]
method = st.sidebar.selectbox("VAD Method", methods)
SAMPLE_WINDOW = st.sidebar.slider("Window Size (s)", 0.01, 0.1, 0.02, 0.005)
SAMPLE_OVERLAP = st.sidebar.slider("Overlap (s)", 0.005, 0.05, 0.01, 0.005)
THRESHOLD = st.sidebar.slider("Voice Energy Threshold", 0.1, 0.9, 0.6, 0.05)

if 'data' in locals() and data is not None:
    if method == "Energy-Based":
        run_energy_vad(data, rate, SAMPLE_WINDOW, SAMPLE_OVERLAP, THRESHOLD)
    elif method == "WebRTC":
        apply_webrtc_vad(data, rate)
    elif method == "Silero":
        run_silero_vad(data, rate)
    elif method == "Wav2Vec2":
        run_wav2vec_vad(data, rate, SAMPLE_WINDOW, SAMPLE_OVERLAP, THRESHOLD)
