from TTS.api import TTS
import sounddevice as sd
import numpy as np
import re

# Initialize the TTS model
model = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")

# Utility to clean text of emojis and unsupported characters
def clean_text(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub('', text).strip()

# TTS speak function with safety checks
def speak(text):
    cleaned = clean_text(text)

    if not cleaned or len(cleaned.split()) == 0:
        print("⚠️ Skipping empty or invalid speech text.")
        return

    print("🔊 Speaking:", cleaned)
    try:
        wav = model.tts(cleaned)
        wav = np.array(wav, dtype=np.float32)
        sd.play(wav, samplerate=22050)
        sd.wait()
    except Exception as e:
        print(f"❌ TTS error: {e}")
