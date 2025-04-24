import numpy as np

def _connect_energy_with_frequencies(data_window, sample_rate=16000):
    fft = np.fft.rfft(data_window)
    freqs = np.fft.rfftfreq(len(data_window), 1/sample_rate)
    power = np.abs(fft) ** 2
    return dict(zip(freqs, power))

def _sum_energy_in_band(energy_dict, start_band=300, end_band=3000):
    return sum(v for k, v in energy_dict.items() if start_band <= k <= end_band)

def _median_filter(x, k):
    assert k % 2 == 1
    assert x.ndim == 1
    k2 = (k - 1) // 2
    y = np.zeros((len(x), k), dtype=x.dtype)
    y[:, k2] = x
    for i in range(k2):
        j = k2 - i
        y[j:, i] = x[:-j]
        y[:j, i] = x[0]
        y[:-j, -(i + 1)] = x[j:]
        y[-j:, -(i + 1)] = x[-1]
    return np.median(y, axis=1)

def _smooth_speech_detection(detected_voice, speech_window=0.5, window=0.02):
    median_window = int(speech_window / window)
    if median_window % 2 == 0:
        median_window -= 1
    detected_voice_array = np.array(detected_voice, dtype=int)
    return _median_filter(detected_voice_array, median_window)

def get_speech_segments(smoothed_voice, window_size=0.02):
    segments = []
    start = None
    for i, val in enumerate(smoothed_voice):
        if val and start is None:
            start = i
        elif not val and start is not None:
            segments.append((start * window_size, i * window_size))
            start = None
    if start is not None:
        segments.append((start * window_size, len(smoothed_voice) * window_size))
    return segments
