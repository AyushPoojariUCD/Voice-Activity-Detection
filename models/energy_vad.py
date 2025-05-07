import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from utils.frequency_energy_utils import (
    _connect_energy_with_frequencies,
    _sum_energy_in_band,
    _smooth_speech_detection,
    get_speech_segments,
)

def run_energy_vad(data, rate, SAMPLE_WINDOW, SAMPLE_OVERLAP, THRESHOLD):
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

    st.subheader("\U0001F4CC Detected Segments")
    for i, (s, e) in enumerate(segments):
        st.markdown(f"**Segment {i+1}: {s:.2f}s - {e:.2f}s**")