# === display.py ===

import tkinter as tk

# Optional: map emoji to label (expand as needed)
EMOJI_LABELS = {
    "😊": "Happy",
    "😢": "Sad",
    "😠": "Angry",
    "😐": "Neutral",
    "🤔": "Thinking",
    "🥳": "Excited",
    "😴": "Tired",
    "😨": "Scared"
}

def show(emoji: str):
    label_text = EMOJI_LABELS.get(emoji, "Unknown")

    window = tk.Tk()
    window.title("EmotiMate Mood Display")
    window.geometry("250x250")
    window.configure(bg="#f8f8ff")  # Light background
    window.resizable(False, False)

    emoji_label = tk.Label(
        window,
        text=emoji,
        font=("Arial", 90),
        bg="#f8f8ff"
    )
    emoji_label.pack(pady=(30, 10))

    text_label = tk.Label(
        window,
        text=f"Mood: {label_text}",
        font=("Helvetica", 16, "bold"),
        fg="#333",
        bg="#f8f8ff"
    )
    text_label.pack()

    # Auto-close after 2 seconds
    window.after(2000, window.destroy)
    window.mainloop()
