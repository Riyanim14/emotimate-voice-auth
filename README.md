# 🎙️ Emotimate Voice Authentication System

This is a Python-based voice authentication system that:

- Detects a wake word (e.g. "Jarvis") using **Porcupine**
- Records speech only when speech is detected using **WebRTC VAD**
- Creates and compares voice embeddings using **Resemblyzer**
- Matches the speaker using **FAISS** vector similarity search

---

## 🔧 Requirements

### 🐍 Python Version
```bash
Python 3.8 - 3.10 (Recommended: 3.10)
```

### 📦 Dependencies

All required dependencies are listed in the `requirements.txt` file. Install them with:

```bash
pip install -r requirements.txt
```

### 📁 .env File

Create a `.env` file in the root directory with the following:

```env
PORCUPINE_KEY=your_porcupine_access_key_here
```

You can get your [Porcupine Access Key](https://console.picovoice.ai/) by signing up at Picovoice.

---

## 🚀 How It Works

1. Listens for the wake word "Jarvis"
2. On detection, records the user's voice
3. Converts the recording into a voice embedding
4. Compares the embedding with stored users using FAISS
5. Authenticates the speaker or prompts to register if unknown

---

## 📂 Folder Structure

```
emotimate_voice_auth/
│
├── user_data/          # Stores voice embeddings per user
├── voice_auth.py       # Main executable
├── .env                # API keys (not committed)
├── requirements.txt    # All dependencies
└── README.md           # Project overview
```

---

## 👤 Adding New Users

If the voice is not recognized, the system prompts to add a new user and stores their embedding.

---

## ✅ Future Improvements

- Add GUI
- Improve multi-user handling
- Support for re-training or deleting users

---

## 🛡️ Disclaimer

This project is a prototype and **not secure** for production-level authentication.
