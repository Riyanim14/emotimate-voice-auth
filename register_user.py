from modules import stt, authentication

if __name__ == "__main__":
    user_id = input("Enter new user ID: ")
    audio = stt.capture_audio()
    authentication.enroll_user(audio, user_id)
    print(f"✅ User '{user_id}' enrolled.")
