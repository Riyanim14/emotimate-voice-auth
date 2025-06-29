# === authentication.py ===

import numpy as np
import faiss
import pickle
import uuid
from pathlib import Path
from resemblyzer import VoiceEncoder, preprocess_wav
import sounddevice as sd
from scipy.io.wavfile import write
from datetime import datetime

# === Constants & Config ===
USER_DIR = Path("database/voice_signatures")
INDEX_PATH = USER_DIR / "faiss.index"
ID_PATH = USER_DIR / "user_ids.pkl"
UNKNOWN_DIR = USER_DIR / "unknown_attempts"

THRESHOLD = 0.6
EMBEDDING_DIM = 256
SAMPLE_RATE = 16000
RECORD_SECONDS = 6

USER_DIR.mkdir(parents=True, exist_ok=True)
UNKNOWN_DIR.mkdir(parents=True, exist_ok=True)

encoder = VoiceEncoder()
index = faiss.IndexFlatL2(EMBEDDING_DIM)
user_ids = []
user_match_count = {}

# === Index Handling ===
def _load_index():
    global index, user_ids
    if INDEX_PATH.exists() and ID_PATH.exists():
        index = faiss.read_index(str(INDEX_PATH))
        with open(ID_PATH, "rb") as f:
            user_ids[:] = pickle.load(f)
    else:
        _rebuild_index()

def _rebuild_index():
    global index, user_ids
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    user_ids.clear()
    embeddings = []
    for file in USER_DIR.glob("*.npy"):
        emb = np.load(file)
        embeddings.append(emb)
        user_ids.append(file.stem)
    if embeddings:
        index.add(np.vstack(embeddings))
    with open(ID_PATH, "wb") as f:
        pickle.dump(user_ids, f)
    faiss.write_index(index, str(INDEX_PATH))

# === Audio Helpers ===
def _record_audio(seconds=RECORD_SECONDS, sample_rate=SAMPLE_RATE):
    print(f"[🎙️] Recording for {seconds} seconds...")
    audio = sd.rec(int(seconds * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    print("[✔️] Recording complete.")
    return np.squeeze(audio)

def _log_unknown_voice(audio, sample_rate=SAMPLE_RATE):
    filename = UNKNOWN_DIR / f"unknown_{uuid.uuid4().hex[:6]}.wav"
    write(filename, sample_rate, audio)
    print(f"[📁] Unrecognized audio saved: {filename}")

# === Enrollment ===
def enroll_user_from_mic(user_id):
    audio = _record_audio()
    wav = preprocess_wav(audio)
    embed = encoder.embed_utterance(wav)
    np.save(USER_DIR / f"{user_id}.npy", embed)
    _rebuild_index()
    print(f"[+] Enrolled user: {user_id}")

# === Matching & Identification ===
def _update_embedding(user_id, new_embedding):
    path = USER_DIR / f"{user_id}.npy"
    if path.exists():
        old_embedding = np.load(path)
        updated = (old_embedding + new_embedding) / 2
        np.save(path, updated)
        _rebuild_index()
        print(f"[🧠] Updated embedding for '{user_id}'")

def _match_user(wav):
    global user_match_count
    embed = encoder.embed_utterance(wav)
    D, I = index.search(embed.reshape(1, -1), 1)
    distance = D[0][0]
    match_idx = I[0][0]

    if match_idx >= len(user_ids):
        _log_unknown_voice(wav)
        print("[❌] Invalid match index.")
        return None

    user_id = user_ids[match_idx]
    similarity = max(0, (1 - distance)) * 100
    print(f"[📊] Candidate: '{user_id}' | Distance: {distance:.2f} | Similarity: {similarity:.1f}%")

    if distance <= THRESHOLD:
        print(f"[✅] Recognized as '{user_id}'")
        user_match_count[user_id] = user_match_count.get(user_id, 0) + 1
        if user_match_count[user_id] >= 5:
            _update_embedding(user_id, embed)
            user_match_count[user_id] = 0
        return user_id

    elif distance <= 0.90:
        print(f"[🤔] Close to '{user_id}' but not confident.")
        choice = input("🔁 Re-register to improve match? (y/n): ").strip().lower()
        if choice == "y":
            enroll_user_from_mic(user_id)
            return user_id
        else:
            return None

    else:
        _log_unknown_voice(wav)
        print(f"[❌] Speaker not recognized (distance {distance:.2f})")
        return None

# === Public Interfaces ===
def identify_user(audio=None):
    _load_index()
    if index.ntotal == 0:
        print("[!] No enrolled users found.")
        return None

    if audio is None:
        audio = _record_audio()
    wav = preprocess_wav(audio)
    return _match_user(wav)

def register_user():
    user_id = input("🆔 Enter name/ID for new user: ").strip()
    if user_id:
        enroll_user_from_mic(user_id)
    else:
        print("⚠️ User ID cannot be empty.")

def auto_register_unknown(audio):
    user_id = input("🆔 Enter a name for the new user: ").strip()
    if not user_id:
        print("⚠️ Registration cancelled. No name entered.")
        return None
    try:
        wav = preprocess_wav(audio)
        embed = encoder.embed_utterance(wav)
        np.save(USER_DIR / f"{user_id}.npy", embed)
        _rebuild_index()
        print(f"[+] Enrolled user: {user_id}")
        return user_id
    except Exception as e:
        print(f"[!] Auto-registration failed: {e}")
        return None

def list_users():
    print("\n👥 Registered Users:")
    if not user_ids:
        _load_index()
    if not user_ids:
        print("No users found.")
    else:
        for idx, uid in enumerate(user_ids, start=1):
            print(f"{idx}. {uid}")
    print()

def delete_user(user_id):
    file_path = USER_DIR / f"{user_id}.npy"
    if file_path.exists():
        file_path.unlink()
        print(f"[🗑️] Deleted user: {user_id}")
        _rebuild_index()
    else:
        print(f"[⚠️] No user found with ID: {user_id}")

def manage_users():
    _load_index()
    while True:
        print("\n==== 🔧 User Management Menu ====")
        print("1. View registered users")
        print("2. Register a new user")
        print("3. Delete a user")
        print("4. Exit")
        choice = input("👉 Choose an option (1-4): ").strip()

        if choice == "1":
            list_users()
        elif choice == "2":
            register_user()
        elif choice == "3":
            uid = input("🆔 Enter user ID to delete: ").strip()
            delete_user(uid)
        elif choice == "4":
            print("👋 Exiting user management.")
            break
        else:
            print("❌ Invalid choice. Try again.")

# === Entry Point ===
if __name__ == "__main__":
    print("\n==== 🔐 Voice Authentication Menu ====")
    print("1. Authenticate a user")
    print("2. Manage users (view, register, delete)")
    print("3. Exit")

    option = input("👉 Choose an option (1-3): ").strip()

    if option == "1":
        identified_user = identify_user()
        if identified_user:
            print(f"\n✅ Welcome, {identified_user}!")
        else:
            print("\n❌ Could not authenticate. Please try again.")
    elif option == "2":
        manage_users()
    else:
        print("👋 Exiting.")
