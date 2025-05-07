import numpy as np

def zcr_vad_segments(audio, rate, window_duration=0.025, threshold=0.4):
    frame_length = int(rate * window_duration)
    num_frames = int(np.ceil(len(audio) / frame_length))

    segments = []
    current_start = None

    for i in range(num_frames):
        start = i * frame_length
        end = start + frame_length
        frame = audio[start:end]
        if len(frame) == 0:
            continue
        zcr = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * len(frame))

        time = i * window_duration
        if zcr > threshold and current_start is None:
            current_start = time
        elif zcr <= threshold and current_start is not None:
            segments.append((current_start, time))
            current_start = None

    if current_start is not None:
        segments.append((current_start, num_frames * window_duration))

    return segments
