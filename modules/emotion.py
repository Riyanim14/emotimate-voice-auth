from pyAudioAnalysis import ShortTermFeatures, audioBasicIO
import numpy as np
import tempfile
from scipy.io.wavfile import write
import os

def detect(audio):
    # Create a named temp file, but close it immediately after getting the name
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name

    try:
        # Write the audio to file AFTER exiting 'with', so it's not locked
        write(temp_path, 16000, audio)

        # Now read and process the file
        fs, signal = audioBasicIO.read_audio_file(temp_path)
        F, _ = ShortTermFeatures.feature_extraction(signal, fs, int(0.05 * fs), int(0.025 * fs))

        energy = np.mean(F[1])
        zcr = np.mean(F[0])

        if energy < 0.01:
            return "sad"
        elif zcr > 0.1:
            return "angry"
        elif energy > 0.2:
            return "happy"
        else:
            return "neutral"

    except Exception as e:
        print(f"[emotion.py] Emotion detection failed: {e}")
        return "neutral"

    finally:
        # Ensure the temp file is always removed
        if os.path.exists(temp_path):
            os.remove(temp_path)
