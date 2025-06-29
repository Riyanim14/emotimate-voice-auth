# === main.py ===

import os
import time
import argparse
import sys
from dotenv import load_dotenv

# Fix potential OpenMP runtime warning
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load environment variables from .env
load_dotenv()

# === Custom Modules ===
from modules import (
    wake_word,
    stt,
    emotion,
    authentication,
    memory,
    core_model,
    tts,
    face_engine,
    display
)

# Define stop keywords for graceful shutdown
STOP_KEYWORDS = {"stop", "exit", "goodbye", "shut down", "terminate"}

def session():
    print("🎧 Voice Assistant Initialized. Listening for wake word...")

    while True:
        if wake_word.listen_for_wake():
            print("🔓 Wake word detected. Starting session...")
            tts.speak("Wake word detected")
            time.sleep(1.5)  # Wait before recording user speech
            start_time = time.time()

            try:
                audio = stt.capture_audio(duration=6)
                text = stt.transcribe(audio)

                if not text:
                    print("⚠️ No speech detected.")
                    tts.speak("I couldn't hear you clearly. Please try again.")
                    continue

                # Check for stop keywords
                if any(word in text.lower() for word in STOP_KEYWORDS):
                    tts.speak("Goodbye! Shutting down.")
                    print("💟 Stop word detected. Terminating the assistant.")
                    sys.exit()

                # Identify speaker
                user_id = authentication.identify_user(audio)
                if not user_id:
                    print("❌ User not identified.")
                    tts.speak("I couldn't verify your identity. Would you like to register?")
                    choice = input("👉 Register this voice? (y/n): ").strip().lower()
                    if choice == 'y':
                        user_id = authentication.auto_register_unknown(audio)
                        if user_id:
                            tts.speak(f"Thanks, {user_id}. You're now registered!")
                        else:
                            tts.speak("Registration failed or was cancelled.")
                        continue
                    else:
                        tts.speak("Okay, maybe next time.")
                        continue

                # Detect emotion
                emo_state = emotion.detect(audio)
                print(f"👤 User: {user_id} | 🧠 Mood: {emo_state} | 📔 Text: {text}")

                # Greet returning user
                memory_data = memory.fetch(user_id)
                if memory_data and "user_name" in memory_data:
                    tts.speak(f"Welcome back, {memory_data['user_name']}!")

                # Generate response
                response = core_model.generate(text, user_id, emo_state, memory_data)
                tts.speak(response)

                # Visual + memory update
                emoji = face_engine.get_emoji(emo_state)
                display.show(emoji)
                memory.update(user_id, text, response)

                elapsed = time.time() - start_time
                print(f"🕒 Session processing time: {elapsed:.2f}s")

            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                tts.speak("Sorry, something went wrong.")

def main():
    print("🚀 main.py started!")

    parser = argparse.ArgumentParser(description="Voice Assistant")
    parser.add_argument('--mode', choices=['run', 'register', 'manage'], default='run', help="Select mode: run / register / manage")
    args = parser.parse_args()

    if args.mode == 'register':
        authentication.register_user()
    elif args.mode == 'manage':
        authentication.manage_users()
    else:
        session()

if __name__ == "__main__":
    main()
