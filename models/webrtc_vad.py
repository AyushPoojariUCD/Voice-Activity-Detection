import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

try:
    import webrtcvad
except ImportError:
    st.warning("WebRTC VAD not installed. Run: pip install webrtcvad")
    webrtcvad = None

def apply_webrtc_vad(data, rate):
    if webrtcvad is None:
        return
    vad = webrtcvad.Vad(3)
    frame_ms = 30
    frame_size = int(rate * frame_ms / 1000)
    segments = []
    start = None
    for i in range(0, len(data) - frame_size, frame_size):
        frame = (data[i:i+frame_size] * 32767).astype(np.int16).tobytes()
        if vad.is_speech(frame, rate):
            if start is None:
                start = i / rate
        else:
            if start is not None:
                end = i / rate
                segments.append((start, end))
                start = None
    if start:
        segments.append((start, len(data) / rate))

    fig, ax = plt.subplots(figsize=(12, 3))
    time = np.arange(len(data)) / rate
    ax.plot(time, data, alpha=0.5)
    for s, e in segments:
        ax.axvspan(s, e, color='red', alpha=0.3)
    st.pyplot(fig)

    st.subheader("\U0001F4CC Detected Segments")
    for i, (s, e) in enumerate(segments):
        st.markdown(f"**Segment {i+1}: {s:.2f}s - {e:.2f}s**")