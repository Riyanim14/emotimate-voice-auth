# Voice Assistant Prototype

### 🧠 Features
- Wake word detection with Porcupine
- STT with Whisper
- Emotion detection with pyAudioAnalysis
- Speaker authentication using Resemblyzer + FAISS
- TTS using Coqui
- Memory system (TinyDB + SQLite)

---

## 🚀 Setup Instructions

```bash
git clone <repo>
cd voice_assistant_project
pip install -r requirements.txt
```

## 👤 Register a user

```bash
python register_user.py
```

## ▶️ Run the assistant

```bash
python main.py
```

Ensure your mic is enabled and accessible.
