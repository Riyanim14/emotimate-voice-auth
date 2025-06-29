import whisper
import sounddevice as sd
import numpy as np

model = whisper.load_model("base")

def capture_audio(duration=6, samplerate=16000):
    print("🎤 Recording...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    audio = np.squeeze(audio)
    return audio

def transcribe(audio):
    print("🧹 Reducing background noise...")
    # Optional: noise reduction
    # import noisereduce as nr
    # audio = nr.reduce_noise(y=audio, sr=16000)

    result = model.transcribe(audio, fp16=False, language="en")
    text = result['text'].strip()
    print(f"📄 Transcribed Text: {text}")
    return text
