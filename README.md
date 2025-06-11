
# EmotiMate Voice-Based User Authentication

A Python-based voice authentication system designed for EmotiMate to detect who is speaking using speaker embeddings. It supports wake-word activation, voice-based user identification, and onboarding of new users — all offline.

---

## 🔧 Features

- 🎤 **Wake Word Detection** ("Jarvis") using [Picovoice Porcupine](https://picovoice.ai/products/porcupine/)
- 🔊 **Voice Recording** using `sounddevice`
- 🧠 **Speaker Embeddings** with `Resemblyzer`
- ⚡ **Fast User Matching** via FAISS (cosine similarity)
- 👤 **New User Onboarding** when no match is found
- 📴 **Offline-first Operation** — suitable for embedded devices

---

## 📁 Folder Structure

```
emotimate_voice_auth/
├── user_data/             # Stored speaker embeddings (.npy)
├── voice_auth.py          # Main script
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
```

---

## 🚀 Quick Start

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

### 2. Set Up Porcupine
- Create an account and get an **Access Key** at: https://console.picovoice.ai/
- Replace `YOUR_ACCESS_KEY` in `voice_auth.py` with your actual key.

### 3. Run the Script

```bash
python voice_auth.py
```

- Say the wake word (default: "Jarvis")
- Speak a short phrase
- The system will recognize your voice or prompt to create a new user profile

---

## 🧠 How It Works

1. **Wake Word Activation**
   - Uses Porcupine to start listening only when triggered

2. **Speaker Embedding**
   - Uses `Resemblyzer` to convert audio into a fixed-length embedding

3. **Fast Matching**
   - Uses `FAISS` to find the closest known voice vector

4. **Onboarding**
   - If no match, prompt for user name and save embedding

---

## 📦 Dependencies

- `resemblyzer`
- `sounddevice`
- `scipy`
- `faiss-cpu`
- `pvporcupine`

---

## 🛠 Future Improvements

- Add support for `pyannote.audio` for speaker diarization (multi-speaker handling)
- Encrypt stored embeddings
- Migrate from `.npy` to SQLite or vector DB for large-scale use
- Add VAD (Voice Activity Detection) to trim silence

---

## 📄 License

MIT License — free to use with attribution.
