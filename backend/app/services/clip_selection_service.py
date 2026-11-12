
import json
import re

from app.services.llm_service import generate_response


SYSTEM_PROMPT = """
You are an expert short-form video editor.

Your job is to identify sections of long-form videos
that could become engaging YouTube Shorts, TikTok videos,
or Instagram Reels.

A good clip should:
1. Make sense on its own.
2. Have a strong hook.
3. Contain a complete idea, story, argument, insight,
   explanation, or punchline.
4. Be interesting, useful, surprising, emotional, or entertaining.
5. Avoid excessive filler.
6. Avoid starting or ending in the middle of a thought.
7. Be between 20 and 60 seconds long.

IMPORTANT:
- Use ONLY the start and end timestamps provided in the input sentences/chunks.
- Never invent timestamps.
- Return ONLY valid JSON.
"""


def build_prompt(chunk_batch: list[tuple[int, dict]]) -> str:
    chunks_text = ""

    for index, chunk in chunk_batch:
        sentences = chunk.get("sentences", [])
        if sentences:
            lines = "\n".join(
                f"[{s['start']:.2f} - {s['end']:.2f}] {s['text']}"
                for s in sentences
            )
        else:
            lines = f"[{chunk['start']:.2f} - {chunk['end']:.2f}] {chunk['text']}"

        chunks_text += f"""
CHUNK {index} (Range: {chunk["start"]:.2f}s to {chunk["end"]:.2f}s)
TRANSCRIPT WITH TIMESTAMPS:
{lines}
------------------------------
"""

    return f"""
Analyze the following transcript chunks.

{chunks_text}

Identify the best segments that work as standalone short-form videos (20 to 60 seconds).
You may combine consecutive sentences to form a complete 20-60s clip using their exact start and end timestamps.

Return ONLY this JSON structure:

{{
    "clips": [
        {{
            "chunk_index": 0,
            "start": 120.5,
            "end": 165.8,
            "score": 8.7,
            "title": "Short descriptive title",
            "hook": "Strong opening idea",
            "reason": "Why this section works as a short-form clip"
        }}
    ]
}}

Rules:
- score must be between 0 and 10.
- Do not include clips with score below 6.
- Clip duration (end - start) MUST be between 15 and 65 seconds.
- Use exact timestamps from the provided transcript lines.
"""


def _extract_json_dict(raw_text: str) -> dict:
    cleaned = raw_text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


import os
from pathlib import Path



def select_clips(chunks: list[dict], cache_path: str = "../data/transcripts/selected_clips.json") -> list[dict]:
    # 1. Check if clips were already selected for this video
    if cache_path and os.path.exists(cache_path) and os.path.getsize(cache_path) > 10:
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if cached:
                print(f"   -> Loaded {len(cached)} selected clips from cache ({cache_path}).")
                return cached
        except Exception:
            pass

    batch_size = 6
    overlap = 1

    all_selected_clips = []
    seen_clips = set()

    indexed_chunks = list(enumerate(chunks))

    i = 0
    while i < len(indexed_chunks):
        batch = indexed_chunks[i:i + batch_size]
        print(f"   -> Processing batch of {len(batch)} chunks (from index {batch[0][0]} to {batch[-1][0]})...")

        prompt = build_prompt(batch)
        response = generate_response(SYSTEM_PROMPT, prompt)

        try:
            result = _extract_json_dict(response)

            for clip in result.get("clips", []):
                start = round(float(clip["start"]), 1)
                end = round(float(clip["end"]), 1)
                if end <= start or (end - start) < 10:
                    continue

                clip_identifier = (start, end)
                if clip_identifier not in seen_clips:
                    seen_clips.add(clip_identifier)
                    clip["start"] = float(clip["start"])
                    clip["end"] = float(clip["end"])
                    all_selected_clips.append(clip)

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"\n   ⚠ Invalid JSON returned by LLM ({e}):")
            print(response[:300])

        i += (batch_size - overlap)
        if i >= len(indexed_chunks) and len(indexed_chunks) > batch_size:
            break

    all_selected_clips.sort(key=lambda x: x.get("score", 0), reverse=True)

    # Save to cache so future runs skip LLM calls!
    if cache_path and all_selected_clips:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(all_selected_clips, f, indent=2, ensure_ascii=False)

    return all_selected_clips






# import json
# import re

# from app.services.llm_service import generate_response


# SYSTEM_PROMPT = """
# You are an expert short-form video editor.

# Your job is to identify sections of long-form videos
# that could become engaging YouTube Shorts, TikTok videos,
# or Instagram Reels.

# A good clip should:

# 1. Make sense on its own.
# 2. Have a strong hook.
# 3. Contain a complete idea, story, argument, insight,
#    explanation, or punchline.
# 4. Be interesting, useful, surprising, emotional, or entertaining.
# 5. Avoid excessive filler.
# 6. Avoid starting or ending in the middle of a thought.
# 7. Prefer clips between 20 and 60 seconds.

# IMPORTANT:

# - Use ONLY timestamps provided in the input.
# - Never invent timestamps.
# - Do not modify timestamps.
# - Return ONLY valid JSON.
# """


# def build_prompt(chunk_batch: list[tuple[int, dict]]) -> str:

#     chunks_text = ""

#     for index, chunk in chunk_batch:

#         chunks_text += f"""
# CHUNK {index}

# START: {chunk["start"]}
# END: {chunk["end"]}

# TRANSCRIPT:
# {chunk["text"]}

# ------------------------------
# """

#     return f"""
# Analyze the following transcript chunks.

# {chunks_text}

# Identify the chunks that could work as short-form videos.

# Return ONLY this JSON structure:

# {{
#     "clips": [
#         {{
#             "chunk_index": 0,
#             "start": 120.5,
#             "end": 165.8,
#             "score": 8.7,
#             "title": "Short descriptive title",
#             "hook": "Strong opening idea",
#             "reason": "Why this section works as a short-form clip"
#         }}
#     ]
# }}

# Rules:

# - score must be between 0 and 10.
# - Do not include clips with score below 6.
# - Prefer 20–60 second clips.
# - Prefer complete thoughts.
# - Prefer strong hooks.
# - Prefer surprising, useful, emotional, educational,
#   controversial, funny, or insightful moments.
# - Do not invent timestamps.
# - Use timestamps from the provided chunks.
# """
    

# def select_clips(chunks: list[dict]) -> list[dict]:
    
#     # We will process chunks in batches to avoid LLM context limits.
#     # We use a batch size of 15, with an overlap of 2 chunks to ensure 
#     # the LLM has context and no clips are left at the boundaries.
#     batch_size = 15
#     overlap = 2
    
#     all_selected_clips = []
#     seen_clips = set() # To filter duplicates caused by overlap

#     indexed_chunks = list(enumerate(chunks))

#     i = 0
#     while i < len(indexed_chunks):
        
#         batch = indexed_chunks[i:i + batch_size]
#         print(f"   -> Processing batch of {len(batch)} chunks (from index {batch[0][0]} to {batch[-1][0]})...")
        
#         prompt = build_prompt(batch)

#         response = generate_response(
#             SYSTEM_PROMPT,
#             prompt
#         )

#         try:
#             # Clean possible markdown wrapping from LLM response
#             cleaned_response = response.replace("```json", "").replace("```", "").strip()
#             result = json.loads(cleaned_response)

#             for clip in result.get("clips", []):
#                 clip_identifier = (clip["start"], clip["end"])
#                 if clip_identifier not in seen_clips:
#                     seen_clips.add(clip_identifier)
#                     all_selected_clips.append(clip)

#         except (json.JSONDecodeError, KeyError):
#             print("\nInvalid JSON returned by LLM:")
#             print(response)

#         # Move forward, accounting for overlap
#         i += (batch_size - overlap)
#         if i >= len(indexed_chunks) and len(indexed_chunks) > batch_size:
#             break
            
#     # Sort clips by score so the best ones are always prioritized
#     all_selected_clips.sort(key=lambda x: x.get("score", 0), reverse=True)
#     return all_selected_clips