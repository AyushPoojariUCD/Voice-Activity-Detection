# 🎤 Voice Activity Detection (VAD) App

This Streamlit application performs **Voice Activity Detection** and **Speech-to-Text transcription** using:

- WebRTC Voice Activity Detection
- Silero Voice Activity Detection
- Entropy based Voice Acitvity Detection
- Energy based Voice Activity Detection
- ZCR based Voice Activity Detection

Supports both **WAV file uploads** and **real-time microphone input**.

---

## 🚀 Features

- 🔊 Upload or record audio
- 🎯 Detect voice activity using frequency-energy analysis
- 🧾 Transcribe audio with Whisper and Wav2Vec2
- 📊 Interactive waveform plots with voice segment overlays
- ⚙️ Adjustable VAD parameters

---

## 📦 Installation

Follow these steps to set up the Voice Activity Detection (VAD) App on your local machine.

### 1. Clone the repository

```bash
git clone https://github.com/AyushPoojariUCD/Voice-Activity-Detection.git
```

### 2. Navigate to the project directory

```
cd Voice-Activity-Detection
```

### 3. Create a virtual environment

```
python -m venv venv
```

### 4. Install dependencies

```
pip install -r requirements.txt
```

#### or

```
pip install numpy scipy scikit-learn librosa streamlit openai torch torchaudio
```

### 5. Run the application

```
streamlit run app.py
```

---

### 6. View notebook:

```
https://colab.research.google.com/drive/1WCm8WwlX0It8BA2r5rbV-WlKJ3munGia?usp=sharing
```

### Key Files and Directories:

- **`app.py`**: The main Streamlit application file where users can interact with the app, upload or record audio, and view the results.
- **`utils/`**: Contains utility functions, including the energy-based VAD algorithm and audio processing utilities.
  - **`frequency_energy_utils.py`**: A file containing functions for audio file processing and frequency-energy analysis for VAD.
- **`models/`**: Contains different machine learning models for future VAD improvements.
  - **`cnn_model.py`**: The file for a Convolutional Neural Network-based VAD model that will be implemented later.
  - **`transformer_model.py`**: A placeholder for a future Transformer-based model for VAD.
- **`data/`**: Stores example datasets, audio samples, and other related resources for testing and validation.
  - **`audio_samples/`**: A directory for storing sample audio files used for testing the VAD.
- **`logs/`**: A placeholder directory for storing logs of predictions, performance metrics, and monitoring results (will be added in the future).
- **`.env`**: Stores environment variables, which will be set up later for sensitive configurations like API keys or model paths.

`This structure ensures that the project is modular, organized, and can easily scale as new features and models are added in the future.`

---

### 📌 Project Code that will be updated

### 1. **Integration with CNN Models**:

`We will integrate **Convolutional Neural Networks (CNNs)** for more accurate voice activity detection and also transformer.`

### 2. **WebRTC Integration**:

`We plan to add **WebRTC** (Web Real-Time Communication) capabilities to capture audio streams directly from web browsers in real-time. This integration will enable seamless VAD for live audio without the need for pre-recorded files.`

### 3. **More VAD will be integrated**:

`We plan to add more VAD for integration and compare them.`

### 4. **Model Comparison**:

`We'll compare the performance of **energy-based VAD**, **CNN models**, **WebRTC**, and **Pico Voice** in terms of **accuracy**, **speed**, and **robustness**.`

---
