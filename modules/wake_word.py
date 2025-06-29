# modules/wake_word.py

import os
import struct
import pvporcupine
import pyaudio
from dotenv import load_dotenv

# ✅ Load environment variables
load_dotenv()

# ✅ Get Picovoice Access Key
access_key = os.getenv("PICOVOICE_KEY")
if not access_key:
    raise ValueError("❌ Missing PICOVOICE_KEY in your .env file.")

# ✅ Custom Wake Word file path
CUSTOM_WAKEWORD_PATH = "Emotion_en_windows_v3_0_0.ppn"

# ✅ Initialize Porcupine
porcupine = pvporcupine.create(
    access_key=access_key,
    keyword_paths=[CUSTOM_WAKEWORD_PATH]
)

# ✅ Initialize PyAudio
pa = pyaudio.PyAudio()
stream = None


def init_stream():
    """Initializes the microphone stream if not already open."""
    global stream
    if stream is None:
        stream = pa.open(
            rate=porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=porcupine.frame_length
        )


def listen_for_wake():
    """Listens for the wake word and returns True if detected."""
    init_stream()
    pcm = stream.read(porcupine.frame_length, exception_on_overflow=False)
    pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)
    return porcupine.process(pcm) >= 0


def cleanup():
    """Stops audio stream and releases resources."""
    global stream
    if stream:
        stream.stop_stream()
        stream.close()
        stream = None
    pa.terminate()
    porcupine.delete()
