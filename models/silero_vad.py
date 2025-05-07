import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wf

try:
    from silero_vad import get_speech_timestamps, read_audio
except ImportError:
    st.warning("Silero VAD not installed. Run: pip install silero-vad")
    get_speech_timestamps, read_audio = None, None

def run_silero_vad(data, rate):
    if get_speech_timestamps is None:
        return
    wf.write("temp.wav", rate, (data * 32767).astype(np.int16))
    wav = read_audio("temp.wav", sampling_rate=rate)
    segments = get_speech_timestamps(wav, sampling_rate=rate)

    fig, ax = plt.subplots(figsize=(12, 3))
    time = np.arange(len(data)) / rate
    ax.plot(time, data, alpha=0.5)
    for seg in segments:
        ax.axvspan(seg['start'] / rate, seg['end'] / rate, color='green', alpha=0.3)
    st.pyplot(fig)

    st.subheader("\U0001F4CC Detected Segments")
    for i, seg in enumerate(segments):
        st.markdown(f"**Segment {i+1}: {seg['start']/rate:.2f}s - {seg['end']/rate:.2f}s**")