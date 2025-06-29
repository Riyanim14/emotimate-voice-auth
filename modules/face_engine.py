def get_emoji(emotion):
    emojis = {
        "happy": "😄",
        "sad": "😢",
        "angry": "😠",
        "neutral": "🙂"
    }
    return emojis.get(emotion, "🙂")
