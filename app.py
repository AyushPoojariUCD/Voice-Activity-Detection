import torch
import whisper
import librosa
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import scipy.io.wavfile as wf
import pyaudio
import sounddevice as sd
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from utils.frequency_energy_utils import (
    _connect_energy_with_frequencies,
    _sum_energy_in_band,
    _smooth_speech_detection,
    get_speech_segments,
)

# ------- Cached Loading the models -----
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

@st.cache_resource
def load_wav2vec2_model():
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    return processor, model

@st.cache_data
def _read_wav(uploaded_file):
    rate, data = wf.read(uploaded_file)
    if len(data.shape) == 2:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    data /= np.max(np.abs(data))
    return rate, data

# Real-time Audio Recording
def record_audio(duration=5, samplerate=16000):
    """Record audio from the microphone for a specified duration."""
    audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    audio_data = audio_data.flatten()
    audio_data /= np.max(np.abs(audio_data))  # Normalize audio
    return samplerate, audio_data

st.set_page_config(page_title="Voice Activity Detection", layout="wide")
st.title("Voice Activity Detection App")

# Choose mode between file upload or microphone input
mode = st.radio("Select Input Mode", ["Upload WAV File", "Real-time Microphone"])

if mode == "Upload WAV File":
    uploaded_file = st.file_uploader("🎵 Upload a WAV file", type=["wav"])

    if uploaded_file:
        rate, data = _read_wav(uploaded_file)
        st.subheader("🔊 Play Audio")
        st.audio(uploaded_file, format='audio/wav', sample_rate=rate)

elif mode == "Real-time Microphone":
    duration = st.slider("Duration of recording (seconds)", 1, 10, 5)
    if st.button("Record Audio"):
        st.write("Recording...")
        rate, data = record_audio(duration=duration)
        st.write("Recording finished!")
        st.audio(data, format='audio/wav', sample_rate=rate)

# Sidebar: Parameters
st.sidebar.header("VAD Parameters")
SAMPLE_WINDOW = st.sidebar.slider("Window Size (s)", 0.01, 0.1, 0.02, 0.005)
SAMPLE_OVERLAP = st.sidebar.slider("Overlap (s)", 0.005, 0.05, 0.01, 0.005)
THRESHOLD = st.sidebar.slider("Voice Energy Threshold", 0.1, 0.9, 0.6, 0.05)

if 'data' in locals() and data is not None:
    # VAD: Detection
    st.subheader("📈 Raw Audio with VAD Overlay")
    fig, ax = plt.subplots(figsize=(12, 3))
    time = np.arange(len(data)) / rate
    ax.plot(time, data, alpha=0.5, label="Waveform")

    detected_voice_energy = []
    SAMPLE_START = 0
    while SAMPLE_START < len(data) - int(SAMPLE_WINDOW * rate):
        start_idx = int(SAMPLE_START)
        end_idx = start_idx + int(SAMPLE_WINDOW * rate)
        window = data[start_idx:end_idx]
        freq_energy = _connect_energy_with_frequencies(window, rate)
        total_energy = sum(freq_energy.values())
        voice_energy = _sum_energy_in_band(freq_energy)
        ratio = voice_energy / total_energy if total_energy > 0 else 0
        detected_voice_energy.append(ratio > THRESHOLD)
        SAMPLE_START += SAMPLE_OVERLAP * rate

    smoothed_energy = _smooth_speech_detection(detected_voice_energy)
    segments = get_speech_segments(smoothed_energy, window_size=SAMPLE_WINDOW)

    for s, e in segments:
        ax.axvspan(s, e, color="orange", alpha=0.3)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.legend()
    st.pyplot(fig)

    # Smoothed signal plot
    st.subheader("📈 Smoothed VAD Signal")
    fig, ax = plt.subplots(figsize=(10, 2))
    times = np.arange(len(smoothed_energy)) * SAMPLE_OVERLAP
    ax.plot(times, smoothed_energy, label="Smoothed VAD")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voice Detected")
    ax.legend()
    st.pyplot(fig)

    # Voice segment output
    st.subheader("📌 Detected Voice Segments")
    for i, (s, e) in enumerate(segments):
        st.markdown(f"**Segment {i+1}:** {s:.2f}s - {e:.2f}s")

    # === Whisper Transcription ===
    st.subheader("📌 OpenAI Whisper VAD")
    try:
        whisper_model = load_whisper_model()
        audio_16k = librosa.resample(data, orig_sr=rate, target_sr=16000).astype(np.float32)
        with st.spinner("Transcribing with Whisper..."):
            result = whisper_model.transcribe(audio_16k)
        st.markdown(f"**Full Text:** {result['text']}")
        if 'segments' in result:
            st.subheader("Voice Segments Detected")
            for seg in result['segments']:
                st.markdown(f"**{seg['start']:.2f}s - {seg['end']:.2f}s**: {seg['text'].strip()}")
    except Exception as e:
        st.error("Failed to transcribe using Whisper.")
        st.code(str(e))

    # === Wav2Vec2 Transcription ===
    st.subheader("📌 Wav2Vec2 VAD")
    try:
        processor, wav2vec_model = load_wav2vec2_model()
        wav2vec_model.eval()

        segment_texts = []
        with st.spinner("Transcribing detected segments with Wav2Vec2..."):
            st.subheader("Voice Segments Detected")
            for i, (start, end) in enumerate(segments):
                start_sample = int(start * rate)
                end_sample = int(end * rate)
                segment_audio = data[start_sample:end_sample]

                if len(segment_audio) == 0:
                    segment_texts.append("No voice detected")
                    continue

                audio_resampled = librosa.resample(segment_audio, orig_sr=rate, target_sr=16000)
                input_values = processor(audio_resampled, sampling_rate=16000, return_tensors="pt").input_values

                with torch.no_grad():
                    logits = wav2vec_model(input_values).logits
                    predicted_ids = torch.argmax(logits, dim=-1)
                    transcription = processor.batch_decode(predicted_ids)[0]

                segment_texts.append(transcription.strip())

        st.subheader("Wav2Vec2 Segmented Transcriptions")
        for i, text in enumerate(segment_texts):
            s, e = segments[i]
            st.markdown(f"**Segment {i+1}: {s:.2f}s - {e:.2f}s**\n\n{text if text else '[No speech detected]'}")

    except Exception as e:
        st.error("Failed to transcribe using Wav2Vec2.")
        st.code(str(e))
