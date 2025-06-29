# === core_model.py ===

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import requests
import re

model_id = "distilgpt2"
print("🔄 Loading distilgpt2 core model...")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

print("✅ distilgpt2 model loaded successfully.")

# Function to clean city names
def clean_city_name(raw):
    raw = raw.lower()
    raw = raw.replace("today", "")
    raw = re.sub(r"[^a-zA-Z\s]", "", raw)  # remove non-letters
    raw = raw.replace("weather in", "")
    raw = raw.replace("how is", "")
    raw = raw.replace("hows", "")
    raw = raw.replace("what's", "")
    raw = raw.replace("whats", "")
    raw = raw.strip()
    return raw.title()

# Weather API call
def get_weather(text):
    try:
        import os
        api_key = os.getenv("OPENAI_API_KEY")
        city_match = re.search(r"weather.*in ([a-zA-Z\s]+)", text.lower())
        city = clean_city_name(city_match.group(1)) if city_match else "Chandigarh"

        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        response = requests.get(url)
        data = response.json()

        if response.status_code == 200:
            temp = data['main']['temp']
            description = data['weather'][0]['description']
            return f"The weather in {city} is currently {temp}°C with {description}."
        else:
            print(f"⚠️ Failed response from API: {data}")
            return "Sorry, I couldn't fetch the weather right now."

    except Exception as e:
        print(f"❌ Weather fetch error: {e}")
        return "Sorry, I couldn't retrieve the weather."

# Main generation logic
def generate(text, user_id, emotion, memory=None):
    try:
        if "weather" in text.lower():
            return get_weather(text)

        prompt = f"""
You are EmotiMate, a helpful and emotionally aware voice assistant.

User ID: {user_id}
Emotion: {emotion}
User said: "{text}"

Respond in 1–2 short, emotionally aware sentences:
- If sad, be comforting.
- If angry, be calm and supportive.
- If happy, match the excitement.
- If neutral, be clear and kind.

Response:"""

        result = generator(
            prompt,
            max_new_tokens=40,
            temperature=0.7,
            repetition_penalty=1.2
        )
        reply = result[0]['generated_text'].split("Response:")[-1].strip()
        return reply

    except Exception as e:
        print(f"❌ Core model error: {e}")
        return "Sorry, I couldn't generate a response."
