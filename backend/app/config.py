import os

from dotenv import load_dotenv


load_dotenv()


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is not set in the environment. "
        "Get a free key at https://console.groq.com/keys"
    )

if not HF_TOKEN:
    raise ValueError(
        "HF_TOKEN is not set in the environment."
    )