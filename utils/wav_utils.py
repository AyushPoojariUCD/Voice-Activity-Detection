import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wf

def _read_wav(uploaded_file):
    rate, data = wf.read(uploaded_file)
    if len(data.shape) == 2:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    data /= np.max(np.abs(data))
    return rate, data

def record_audio(duration=5, samplerate=16000):
    audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    audio_data = audio_data.flatten()
    audio_data /= np.max(np.abs(audio_data))
    return samplerate, audio_data