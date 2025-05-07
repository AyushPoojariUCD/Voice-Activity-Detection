import streamlit as st
import numpy as np
import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from utils.frequency_energy_utils import (
    _connect_energy_with_frequencies,
    _sum_energy_in_band,
    _smooth_speech_detection,
    get_speech_segments,
)

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
model.eval()

def run_wav2vec_vad(data, rate, SAMPLE_WINDOW, SAMPLE_OVERLAP, THRESHOLD):
    detected = []
    start = 0
    while start < len(data) - int(SAMPLE_WINDOW * rate):
        idx = int(start)
        window = data[idx:idx + int(SAMPLE_WINDOW * rate)]
        freq_energy = _connect_energy_with_frequencies(window, rate)
        total_energy = sum(freq_energy.values())
        voice_energy = _sum_energy_in_band(freq_energy)
        ratio = voice_energy / total_energy if total_energy > 0 else 0
        detected.append(ratio > THRESHOLD)
        start += SAMPLE_OVERLAP * rate

    smoothed = _smooth_speech_detection(detected)
    segments = get_speech_segments(smoothed, window_size=SAMPLE_WINDOW)

    st.subheader("\U0001F9E0 Transcriptions")
    for i, (s, e) in enumerate(segments):
        start_idx = int(s * rate)
        end_idx = int(e * rate)
        segment = data[start_idx:end_idx]
        audio = librosa.resample(segment, orig_sr=rate, target_sr=16000)
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt").input_values
        with torch.no_grad():
            logits = model(inputs).logits
            ids = torch.argmax(logits, dim=-1)
            text = processor.batch_decode(ids)[0]
        st.markdown(f"**Segment {i+1}: {s:.2f}s - {e:.2f}s**")
        st.code(text.strip() if text else "[No speech detected]")