import os
import re
import time
# pyrefly: ignore [missing-import]
import requests
from dotenv import load_dotenv

load_dotenv()

# Current active 2026 Groq production models (each has its own independent TPM bucket)
GROQ_MODELS = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
]
_groq_idx = 0


def _generate_via_groq(
    system_prompt: str,
    user_prompt: str,
    api_key: str
) -> str:
    global _groq_idx
    clean_key = api_key.strip().strip('"').strip("'")
    model_id = GROQ_MODELS[_groq_idx % len(GROQ_MODELS)]

    # Do NOT pass response_format={"type": "json_object"} on gpt-oss/qwen reasoning models,
    # as their internal reasoning tokens trigger Groq's 400 json_validate_failed error.
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 4096,
    }

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {clean_key}",
            "Content-Type": "application/json"
        },
        json=payload,
        timeout=60
    )

    # If model isn't available on this account, rotate to next model in GROQ_MODELS
    if response.status_code == 404:
        _groq_idx = (_groq_idx + 1) % len(GROQ_MODELS)
        raise RuntimeError(f"Groq 404 on {model_id}, switching to {GROQ_MODELS[_groq_idx]}")

    # If TPM rate limit (429) is hit, wait the exact seconds Groq asks for or switch models
    if response.status_code == 429:
        wait_match = re.search(r"try again in ([0-9.]+)s", response.text)
        wait_sec = float(wait_match.group(1)) + 1.5 if wait_match else 15.0
        _groq_idx = (_groq_idx + 1) % len(GROQ_MODELS)
        print(f"   ⏳ Groq TPM limit on {model_id}. Rotating to {GROQ_MODELS[_groq_idx]} after {min(wait_sec, 8.0):.1f}s...")
        time.sleep(min(wait_sec, 8.0))
        raise RuntimeError(f"Groq 429 on {model_id}, rotated to {GROQ_MODELS[_groq_idx]}")

    if not response.ok:
        raise RuntimeError(f"Groq API {response.status_code} ({model_id}): {response.text[:300]}")

    content = response.json()["choices"][0]["message"]["content"] or ""
    # Strip any <think>...</think> blocks before returning JSON
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    return content


def _generate_via_gemini(
    system_prompt: str,
    user_prompt: str,
    api_key: str
) -> str:
    clean_key = api_key.strip().strip('"').strip("'")
    model_name = "gemini-3.8-flash"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"

    payload = {
        "systemInstruction": {
            "parts": [{"text": system_prompt}]
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_prompt}]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json"
        }
    }

    response = requests.post(
        url,
        params={"key": clean_key},
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=90
    )

    if not response.ok:
        raise RuntimeError(f"Gemini API {response.status_code} ({model_name}): {response.text[:250]}")

    data = response.json()
    candidates = data.get("candidates", [])
    if not candidates:
        raise RuntimeError(f"Gemini returned no candidates: {data}")

    parts = candidates[0].get("content", {}).get("parts", [])
    return "".join(part.get("text", "") for part in parts if "text" in part).strip()


def generate_response(
    system_prompt: str,
    user_prompt: str,
    max_retries: int = 5
) -> str:
    groq_key = os.getenv("GROQ_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")

    for attempt in range(1, max_retries + 1):
        try:
            # 1. Use Groq first (ultra-fast ~2s inference on gpt-oss-20b / gpt-oss-120b)
            if groq_key:
                return _generate_via_groq(system_prompt, user_prompt, groq_key)
            # 2. Otherwise use gemini-3.8-flash
            if gemini_key:
                return _generate_via_gemini(system_prompt, user_prompt, gemini_key)
            raise ValueError("Neither GROQ_API_KEY nor GEMINI_API_KEY is set in .env!")

        except Exception as e:
            print(f"   ⚠ Attempt {attempt}/{max_retries}: {e}")
            # If Groq hits repeated rate limits, try Gemini 3.8 Flash on even attempts
            if gemini_key and groq_key and attempt >= 2:
                try:
                    print("   -> Trying gemini-3.8-flash...")
                    return _generate_via_gemini(system_prompt, user_prompt, gemini_key)
                except Exception as gem_err:
                    print(f"   ⚠ Gemini backup notice: {gem_err}")

            if attempt == max_retries:
                raise
            time.sleep(2 * attempt)

    return ""









# import os
# import re
# import time
# # pyrefly: ignore [missing-import]
# import requests
# # pyrefly: ignore [missing-import]
# from huggingface_hub import InferenceClient

# from app.config import HF_TOKEN


# # Ordered from fastest/highest-TPM non-reasoning models to reasoning models
# PREFERRED_GROQ_MODELS = [
#     "llama-3.3-70b-versatile",
#     "llama-3.1-8b-instant",
#     "meta-llama/llama-4-scout-17b-16e-instruct",
#     "meta-llama/llama-4-maverick-17b-128e-instruct",
#     "gemma2-9b-it",
#     "openai/gpt-oss-20b",
#     "openai/gpt-oss-120b",
#     "qwen/qwen3-32b",
# ]

# HF_MODEL = "Qwen/Qwen2.5-7B-Instruct"
# _available_groq_models: list[str] = []
# _current_model_idx: int = 0


# def _get_groq_model_pool(api_key: str) -> list[str]:
#     """
#     Fetches all active chat models on your Groq account and orders them
#     by preference so we can rotate across models if one hits a TPM rate limit.
#     """
#     global _available_groq_models
#     if _available_groq_models:
#         return _available_groq_models

#     clean_key = api_key.strip().strip('"').strip("'")
#     resp = requests.get(
#         "https://api.groq.com/openai/v1/models",
#         headers={"Authorization": f"Bearer {clean_key}"},
#         timeout=15
#     )
#     if not resp.ok:
#         raise RuntimeError(f"Groq /models error ({resp.status_code}): {resp.text}")

#     available_ids = {
#         m["id"] for m in resp.json().get("data", [])
#         if not any(skip in m["id"].lower() for skip in ("whisper", "tts", "guard", "allam", "playai"))
#     }

#     ordered = [m for m in PREFERRED_GROQ_MODELS if m in available_ids]
#     for m in sorted(available_ids):
#         if m not in ordered:
#             ordered.append(m)

#     if not ordered:
#         raise RuntimeError("No active chat models found on Groq account.")

#     _available_groq_models = ordered
#     print(f"   -> Available Groq LLM pool: {_available_groq_models[:4]}")
#     return _available_groq_models


# def _generate_via_groq(
#     system_prompt: str,
#     user_prompt: str,
#     api_key: str
# ) -> str:
#     global _current_model_idx
#     clean_key = api_key.strip().strip('"').strip("'")
#     pool = _get_groq_model_pool(clean_key)

#     model_id = pool[_current_model_idx % len(pool)]
#     is_reasoning_model = any(k in model_id.lower() for k in ("gpt-oss", "qwen3", "qwq", "r1"))

#     payload = {
#         "model": model_id,
#         "messages": [
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_prompt}
#         ],
#         "temperature": 0.2,
#         # Reasoning models need higher token limits so thinking doesn't truncate JSON
#         "max_tokens": 4096 if is_reasoning_model else 2048,
#     }

#     # Only attach strict json_object mode for non-reasoning models
#     if not is_reasoning_model:
#         payload["response_format"] = {"type": "json_object"}

#     response = requests.post(
#         "https://api.groq.com/openai/v1/chat/completions",
#         headers={
#             "Authorization": f"Bearer {clean_key}",
#             "Content-Type": "application/json"
#         },
#         json=payload,
#         timeout=90
#     )

#     # 1. Handle JSON mode validation failure by retrying without response_format
#     if response.status_code == 400 and (
#         "response_format" in response.text or "json_validate_failed" in response.text
#     ):
#         payload.pop("response_format", None)
#         response = requests.post(
#             "https://api.groq.com/openai/v1/chat/completions",
#             headers={
#                 "Authorization": f"Bearer {clean_key}",
#                 "Content-Type": "application/json"
#             },
#             json=payload,
#             timeout=90
#         )

#     # 2. Handle 429 Rate Limit: rotate to next model or parse Groq's exact wait timer
#     if response.status_code == 429:
#         _current_model_idx = (_current_model_idx + 1) % len(pool)
#         next_model = pool[_current_model_idx]

#         # Extract "Please try again in 18.7425s" if present
#         wait_match = re.search(r"try again in ([0-9.]+)s", response.text)
#         wait_hint = float(wait_match.group(1)) if wait_match else 15.0

#         if len(pool) > 1:
#             print(f"   ⚠ TPM limit hit on `{model_id}`. Switching to `{next_model}`...")
#         else:
#             print(f"   ⏳ Rate limit reached. Waiting {wait_hint:.1f}s as requested by Groq...")
#             time.sleep(wait_hint + 1.0)

#         raise RuntimeError(f"Groq API 429 ({model_id}): rotating to {next_model}")

#     if not response.ok:
#         raise RuntimeError(
#             f"Groq API {response.status_code} ({model_id}): {response.text}"
#         )

#     content = response.json()["choices"][0]["message"]["content"] or ""
#     # Strip <think>...</think> blocks if a reasoning model was used
#     content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
#     return content


# def generate_response(
#     system_prompt: str,
#     user_prompt: str,
#     max_retries: int = 5
# ) -> str:
#     groq_key = os.getenv("GROQ_API_KEY")

#     for attempt in range(1, max_retries + 1):
#         try:
#             if groq_key:
#                 return _generate_via_groq(system_prompt, user_prompt, groq_key)

#             client = InferenceClient(api_key=HF_TOKEN)
#             response = client.chat.completions.create(
#                 model=HF_MODEL,
#                 messages=[
#                     {"role": "system", "content": system_prompt},
#                     {"role": "user", "content": user_prompt}
#                 ],
#                 temperature=0.2,
#                 max_tokens=2000
#             )
#             return response.choices[0].message.content or ""

#         except Exception as e:
#             if attempt == max_retries:
#                 raise
#             # Short pause when switching models, longer pause otherwise
#             wait_sec = 2 if "rotating to" in str(e) else min(20, 3 * attempt)
#             print(f"   ⚠ Retry {attempt}/{max_retries} in {wait_sec}s...")
#             time.sleep(wait_sec)

#     return ""









# import time
# # pyrefly: ignore [missing-import]
# from huggingface_hub import InferenceClient

# from app.config import HF_TOKEN


# MODEL = "Qwen/Qwen2.5-7B-Instruct"

# client = InferenceClient(
#     api_key=HF_TOKEN
# )


# def generate_response(
#     system_prompt: str,
#     user_prompt: str,
#     max_retries: int = 3
# ) -> str:
#     for attempt in range(1, max_retries + 1):
#         try:
#             response = client.chat.completions.create(
#                 model=MODEL,
#                 messages=[
#                     {
#                         "role": "system",
#                         "content": system_prompt
#                     },
#                     {
#                         "role": "user",
#                         "content": user_prompt
#                     }
#                 ],
#                 temperature=0.2,
#                 max_tokens=2000
#             )
#             return response.choices[0].message.content or ""

#         except Exception as e:
#             if attempt == max_retries:
#                 raise
#             wait_sec = 2 ** attempt
#             print(f"   ⚠ LLM API error (attempt {attempt}/{max_retries}): {e}. Retrying in {wait_sec}s...")
#             time.sleep(wait_sec)

#     return ""






# # pyrefly: ignore [missing-import]
# from huggingface_hub import InferenceClient

# from app.config import HF_TOKEN


# MODEL = "Qwen/Qwen2.5-7B-Instruct"


# client = InferenceClient(
#     api_key=HF_TOKEN
# )


# def generate_response(
#     system_prompt: str,
#     user_prompt: str
# ) -> str:

#     response = client.chat.completions.create(
#         model=MODEL,
#         messages=[
#             {
#                 "role": "system",
#                 "content": system_prompt
#             },
#             {
#                 "role": "user",
#                 "content": user_prompt
#             }
#         ],
#         temperature=0.2,
#         max_tokens=2000
#     )

#     return response.choices[0].message.content