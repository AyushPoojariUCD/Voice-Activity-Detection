# 🎤 Voice Activity Detection (VAD) App

This Streamlit application performs **Voice Activity Detection** and **Speech-to-Text transcription** using:

- 🧠 OpenAI's Whisper
- 🧠 Facebook's Wav2Vec2
- 📈 Energy-based VAD with visualizations

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


### 📂 Project Structure

Voice-Activity-Detection/
├── README.md                      # Project overview and setup guide
├── app.py                         # Streamlit app for Voice Activity Detection
├── requirements.txt               # List of dependencies
├── utils/                         # Utility functions
│   ├── __init__.py
│   └── frequency_energy_utils.py  # Functions for processing audio files
├── models/                        # Directory for models
│   ├── __init__.py
│   └── cnn_model.py               # VAD ML-based model will be added later
|   |__ transformer_model          # VAD Transformer based model will be added later
|
├── data/                          # Directory for storing datasets
│   └── audio_samples/             # Example audio samples for testing
├── logs/                          # Logs of predictions, performance metrics and monitoring application will be added later
└── .env                           # Environment variables will be added later


### 📌 Project Code that will be updated

### 1. **Integration with CNN Models**:
We will integrate **Convolutional Neural Networks (CNNs)** for more accurate voice activity detection and also transformer. 

### 2. **WebRTC Integration**:
We plan to add **WebRTC** (Web Real-Time Communication) capabilities to capture audio streams directly from web browsers in real-time. This integration will enable seamless VAD for live audio without the need for pre-recorded files.

### 3. **More VAD will be integrated**:
We plan to add more VAD for integration and compare them.

### 4. **Model Comparison**:
We'll compare the performance of **energy-based VAD**, **CNN models**, **WebRTC**, and **Pico Voice** in terms of **accuracy**, **speed**, and **robustness**. This comparative study will help identify the most effective VAD approach for different applications and use cases.
