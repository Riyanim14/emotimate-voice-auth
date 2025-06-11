
import os
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from resemblyzer import VoiceEncoder, preprocess_wav
from resemblyzer.hparams import sampling_rate
from pathlib import Path
import struct
import pvporcupine
import faiss

# Directory for user voice embeddings
USER_DIR = Path("user_data")
USER_DIR.mkdir(exist_ok=True)

encoder = VoiceEncoder()

# Wake Word Detection using Porcupine
def listen_for_wake_word(keyword="jarvis", access_key="YOUR_ACCESS_KEY"):
    porcupine = pvporcupine.create(keywords=[keyword], access_key=access_key)
    print(f"🎧 Listening for wake word: '{keyword}'...")

    with sd.InputStream(channels=1, samplerate=porcupine.sample_rate,
                        blocksize=porcupine.frame_length, dtype='int16') as stream:
        while True:
            pcm = stream.read(porcupine.frame_length)[0]
            pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)
            result = porcupine.process(pcm)
            if result >= 0:
                print("🟢 Wake word detected!")
                break

    porcupine.delete()

# Record voice sample
def record_voice(seconds=4, filename="temp.wav"):
    print(f"[🎤] Recording for {seconds} seconds...")
    audio = sd.rec(int(sampling_rate * seconds), samplerate=sampling_rate, channels=1)
    sd.wait()
    write(filename, sampling_rate, audio)
    print("[✅] Recording complete.")
    return filename

# Generate embedding
def generate_embedding(wav_path):
    wav = preprocess_wav(wav_path)
    embed = encoder.embed_utterance(wav)
    return embed

# Load all saved embeddings
def load_user_embeddings():
    embeddings = {}
    for file in USER_DIR.glob("*.npy"):
        user_id = file.stem
        embeddings[user_id] = np.load(file)
    return embeddings

# Save new user profile
def save_new_user(embedding, username):
    filepath = USER_DIR / f"{username}.npy"
    np.save(filepath, embedding)
    print(f"[💾] New user '{username}' saved!")

# Create FAISS index for fast similarity search
def create_faiss_index(user_embeddings):
    dim = next(iter(user_embeddings.values())).shape[0]
    index = faiss.IndexFlatIP(dim)
    ids = []
    vectors = []

    for user_id, emb in user_embeddings.items():
        ids.append(user_id)
        vectors.append(emb / np.linalg.norm(emb))

    index.add(np.array(vectors).astype('float32'))
    return index, ids

# Search for closest match
def find_match_faiss(embed, faiss_index, user_ids, threshold=0.75):
    norm_embed = embed / np.linalg.norm(embed)
    D, I = faiss_index.search(np.array([norm_embed]).astype('float32'), k=1)
    score = D[0][0]
    idx = I[0][0]

    if score > threshold:
        return user_ids[idx], score
    return None, score

# Main workflow
def main():
    listen_for_wake_word(keyword="jarvis", access_key="YOUR_ACCESS_KEY")

    wav_file = record_voice()
    embed = generate_embedding(wav_file)
    user_embeddings = load_user_embeddings()

    index, ids = create_faiss_index(user_embeddings)
    user_id, score = find_match_faiss(embed, index, ids)

    if user_id:
        print(f"\n✅ Recognized as: {user_id} (Score: {score:.2f})")
    else:
        print("\n❓ New speaker detected.")
        new_user = input("Enter new username: ")
        save_new_user(embed, new_user)

if __name__ == "__main__":
    main()
