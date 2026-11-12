# import subprocess
# from pathlib import Path


# def extract_audio(video_path: str, audio_path: str) -> None:
#     """
#     Extract mono 16 kHz PCM audio from a video.
#     """

#     Path(audio_path).parent.mkdir(
#         parents=True,
#         exist_ok=True
#     )

#     command = [
#         "ffmpeg",
#         "-y",
#         "-i", video_path,
#         "-vn",
#         "-ac", "1",
#         "-ar", "16000",
#         "-c:a", "pcm_s16le",
#         audio_path
#     ]

#     subprocess.run(
#         command,
#         check=True
#     )



import subprocess
from pathlib import Path


def extract_audio(video_path: str, audio_path: str) -> None:
    """
    Extract mono 16 kHz PCM audio with dynamic audio normalization (dynaudnorm)
    so whispered movie dialogue is never missed by Whisper transcription.
    """
    Path(audio_path).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    command = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-i", video_path,
        "-vn",
        "-af", "dynaudnorm=f=150:g=15",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        "-threads", "2",
        audio_path
    ]

    subprocess.run(
        command,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE
    )