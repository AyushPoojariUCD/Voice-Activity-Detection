import numpy as np
from scipy.fft import fft
from scipy.signal import get_window

def spectral_entropy(frame, eps=1e-8):
    spectrum = np.abs(fft(frame))[:len(frame)//2]
    psd = spectrum ** 2
    psd_norm = psd / (np.sum(psd) + eps)
    entropy = -np.sum(psd_norm * np.log2(psd_norm + eps))
    return entropy / np.log2(len(psd_norm))

def entropy_vad_segments(audio, rate, window_duration=0.025, threshold=0.7):
    frame_length = int(rate * window_duration)
    num_frames = int(np.ceil(len(audio) / frame_length))
    window = get_window('hann', frame_length, fftbins=True)

    segments = []
    current_start = None

    for i in range(num_frames):
        start = i * frame_length
        end = start + frame_length
        frame = audio[start:end]
        if len(frame) < frame_length:
            frame = np.pad(frame, (0, frame_length - len(frame)))
        frame = frame * window
        entropy = spectral_entropy(frame)

        time = i * window_duration
        if entropy > threshold and current_start is None:
            current_start = time
        elif entropy <= threshold and current_start is not None:
            segments.append((current_start, time))
            current_start = None

    if current_start is not None:
        segments.append((current_start, num_frames * window_duration))

    return segments
