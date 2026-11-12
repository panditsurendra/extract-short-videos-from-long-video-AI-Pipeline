
import json


def load_transcript(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def extract_words(transcript: dict) -> list[dict]:
    words = []

    for segment in transcript.get("segments", []):
        for word in segment.get("words", []):
            if "start" not in word or "end" not in word:
                continue

            cleaned = word["word"].strip()
            if not cleaned:
                continue

            words.append({
                "word": cleaned,
                "start": float(word["start"]),
                "end": float(word["end"])
            })

    return words


def build_sentences(
    words: list[dict],
    max_words_per_sentence: int = 25
) -> list[dict]:
    sentences = []
    current_words = []

    for word in words:
        current_words.append(word)

        is_punct = word["word"].endswith((".", "?", "!", "।"))
        is_too_long = (
            len(current_words) >= max_words_per_sentence
            and word["word"].endswith((",", ";", ":"))
        ) or len(current_words) >= (max_words_per_sentence + 10)

        if is_punct or is_too_long:
            sentences.append({
                "text": " ".join(w["word"] for w in current_words),
                "start": current_words[0]["start"],
                "end": current_words[-1]["end"]
            })
            current_words = []

    if current_words:
        sentences.append({
            "text": " ".join(w["word"] for w in current_words),
            "start": current_words[0]["start"],
            "end": current_words[-1]["end"]
        })

    return sentences











# # Its job is not transcription.

# # It will only read the existing:

# # aligned_transcript.json

# # and convert:

# # word-level transcript
# #         ↓
# # sentences
# #         ↓
# # candidate chunks



# import json


# def load_transcript(path: str) -> dict:
#     with open(path, "r", encoding="utf-8") as file:
#         return json.load(file)


# def extract_words(transcript: dict) -> list[dict]:
#     words = []

#     for segment in transcript["segments"]:

#         for word in segment.get("words", []):

#             if "start" not in word or "end" not in word:
#                 continue

#             words.append({
#                 "word": word["word"].strip(),
#                 "start": word["start"],
#                 "end": word["end"]
#             })

#     return words






# def build_sentences(words: list[dict]) -> list[dict]:

#     sentences = []
#     current_words = []

#     for word in words:

#         current_words.append(word)

#         text = " ".join(
#             w["word"] for w in current_words
#         )

#         # Simple sentence boundary (English and Hindi)
#         if word["word"].endswith((".", "?", "!", "।")):

#             sentences.append({
#                 "text": text,
#                 "start": current_words[0]["start"],
#                 "end": current_words[-1]["end"],
#                 "words": current_words.copy()
#             })

#             current_words = []

#     # Handle remaining words
#     if current_words:

#         sentences.append({
#             "text": " ".join(
#                 w["word"] for w in current_words
#             ),
#             "start": current_words[0]["start"],
#             "end": current_words[-1]["end"],
#             "words": current_words.copy()
#         })

#     return sentences