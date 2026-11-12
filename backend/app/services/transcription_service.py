import gc
import json
import os
import subprocess
from pathlib import Path

# pyrefly: ignore [missing-import]
import whisperx


def _transcribe_via_groq(audio_path: str, output_path: str, api_key: str, language: str = "en") -> None:
    """
    Zero-CPU cloud transcription using Groq's whisper-large-v3-turbo API.
    Automatically compresses WAV to small MP3 so 30+ min files fit under API limits.
    """
    from huggingface_hub.utils import get_session

    temp_mp3 = str(Path(audio_path).with_suffix(".mp3"))
    subprocess.run([
        "ffmpeg", "-nostdin", "-y", "-i", audio_path,
        "-ac", "1", "-ar", "16000", "-b:a", "32k", temp_mp3
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print("Uploading compressed audio to Groq Whisper API (zero local CPU load)...")
    session = get_session()
    with open(temp_mp3, "rb") as f:
        response = session.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": (Path(temp_mp3).name, f, "audio/mpeg")},
            data={
                "model": "whisper-large-v3-turbo",
                "language": language,
                "response_format": "verbose_json",
                "timestamp_granularities[]": ["word", "segment"]
            },
            timeout=300
        )
    try:
        os.remove(temp_mp3)
    except OSError:
        pass

    response.raise_for_status()
    data = response.json()

    # Format into WhisperX-compatible structure: {"segments": [{"words": [...]}]}
    words = data.get("words", [])
    aligned_result = {"segments": [{"words": words}]}

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(aligned_result, file, indent=2, ensure_ascii=False)

    print(f"Transcript saved to: {output_path}")


def transcribe_and_align(
    audio_path: str,
    output_path: str,
    language: str = "en",
    model_size: str = "base"
) -> None:
    """
    Transcribe audio and obtain accurate word-level timestamps.
    Uses Groq API if GROQ_API_KEY is present in environment; otherwise runs
    memory-optimized local WhisperX with garbage collection between stages.
    """
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        try:
            _transcribe_via_groq(audio_path, output_path, groq_key, language)
            return
        except Exception as e:
            print(f"Groq API failed ({e}), falling back to local WhisperX...")

    device = "cpu"
    cpu_threads = min(4, os.cpu_count() or 4)

    # -----------------------------
    # 1. Load Whisper model
    # -----------------------------
    print(f"Loading Whisper model ('{model_size}', {cpu_threads} threads)...")
    model = whisperx.load_model(
        model_size,
        device=device,
        compute_type="int8",
        threads=cpu_threads
    )

    # -----------------------------
    # 2. Load audio & Transcribe
    # -----------------------------
    audio = whisperx.load_audio(audio_path)

    print("Running Whisper...")
    result = model.transcribe(
        audio,
        language=language,
        batch_size=8
    )

    # Free Whisper model from RAM BEFORE loading alignment model!
    del model
    gc.collect()

    # -----------------------------
    # 3. Load alignment model & Align
    # -----------------------------
    print("Loading alignment model...")
    align_model, metadata = whisperx.load_align_model(
        language_code=language,
        device=device
    )

    print("Aligning words...")
    aligned_result = whisperx.align(
        result["segments"],
        align_model,
        metadata,
        audio,
        device,
        return_char_alignments=False
    )

    # Free alignment model and audio buffer immediately
    del align_model, metadata, audio
    gc.collect()

    # -----------------------------
    # 4. Save result
    # -----------------------------
    Path(output_path).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            aligned_result,
            file,
            indent=2,
            ensure_ascii=False
        )

    print(f"Transcript saved to: {output_path}")











# import json
# from pathlib import Path


# # pyrefly: ignore [missing-import]
# import whisperx


# def transcribe_and_align(
#     audio_path: str,
#     output_path: str,
#     language: str = "en"
# ) -> None:
#     """
#     Transcribe audio using WhisperX and align the transcript
#     to obtain accurate word-level timestamps.
#     """

#     device = "cpu"

#     # -----------------------------
#     # 1. Load Whisper model
#     # -----------------------------

#     print("Loading Whisper model...")

#     model = whisperx.load_model(
#         "medium",
#         device=device,
#         compute_type="int8"
#     )

#     # -----------------------------
#     # 2. Load audio
#     # -----------------------------

#     audio = whisperx.load_audio(
#         audio_path
#     )

#     # -----------------------------
#     # 3. Transcription
#     # -----------------------------

#     print("Running Whisper...")

#     result = model.transcribe(
#         audio,
#         language=language,
#         batch_size=4
#     )

#     # -----------------------------
#     # 4. Load alignment model
#     # -----------------------------

#     print("Loading alignment model...")

#     align_model, metadata = whisperx.load_align_model(
#         language_code=language,
#         device=device
#     )

#     # -----------------------------
#     # 5. Word-level alignment
#     # -----------------------------

#     print("Aligning words...")

#     aligned_result = whisperx.align(
#         result["segments"],
#         align_model,
#         metadata,
#         audio,
#         device,
#         return_char_alignments=False
#     )

#     # -----------------------------
#     # 6. Save result
#     # -----------------------------

#     Path(output_path).parent.mkdir(
#         parents=True,
#         exist_ok=True
#     )

#     with open(
#         output_path,
#         "w",
#         encoding="utf-8"
#     ) as file:

#         json.dump(
#             aligned_result,
#             file,
#             indent=2,
#             ensure_ascii=False
#         )

#     print(f"Transcript saved to: {output_path}")