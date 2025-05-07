import torch
import scipy.io.wavfile as wf  # Corrected the import
from silero_vad import get_speech_timestamps, read_audio
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def run_silero_vad(data, rate):
    # Load the Silero VAD model correctly (unpack the tuple)
    model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad", source="github")
    
    # Save the audio data to a temporary file
    wf.write("temp.wav", rate, (data * 32767).astype(np.int16))
    
    # Read the audio using the appropriate function
    wav = read_audio("temp.wav", sampling_rate=rate)
    
    # Get the speech segments
    segments = get_speech_timestamps(wav, model, sampling_rate=rate)

    # Create a plot to visualize the segments
    fig, ax = plt.subplots(figsize=(12, 3))
    time = np.arange(len(data)) / rate
    ax.plot(time, data, alpha=0.5)
    
    # Highlight the detected segments
    for seg in segments:
        ax.axvspan(seg['start'] / rate, seg['end'] / rate, color='green', alpha=0.3)
    
    # Display the plot using Streamlit
    st.pyplot(fig)

    # Display the detected speech segments
    st.subheader("\U0001F4CC Detected Segments")
    for i, seg in enumerate(segments):
        st.markdown(f"**Segment {i+1}: {seg['start']/rate:.2f}s - {seg['end']/rate:.2f}s**")
