



import os
import platform
import subprocess
from pathlib import Path
import cv2

from app.services.face_tracking_service import get_smooth_crop_centers
from app.services.caption_service import (
    build_caption_events,
    render_caption_on_frame
)

TAIL_PADDING_SEC = 0.7


def _get_video_encoder_args() -> list[str]:
    """
    Uses Apple Silicon / macOS hardware acceleration (h264_videotoolbox)
    to encode in hardware with near-zero CPU heat; falls back to libx264 veryfast.
    """
    if platform.system() == "Darwin":
        try:
            res = subprocess.run(
                ["ffmpeg", "-hide_banner", "-encoders"],
                capture_output=True, text=True, check=True
            )
            if "h264_videotoolbox" in res.stdout:
                return ["-c:v", "h264_videotoolbox", "-b:v", "5500k", "-pix_fmt", "yuv420p"]
        except Exception:
            pass

    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-threads", "4", "-pix_fmt", "yuv420p"]


def _build_audio_mute_filter(mute_intervals: list[tuple[float, float]]) -> list[str]:
    """
    Builds an FFmpeg audio filter (-af) that dips volume to 0 during severe slurs.
    """
    if not mute_intervals:
        return ["-c:a", "copy"]

    filters = [
        f"volume=enable='between(t,{m_start:.3f},{m_end:.3f})':volume=0"
        for m_start, m_end in mute_intervals
    ]
    return ["-af", ",".join(filters), "-c:a", "aac", "-b:a", "128k"]


def _resolve_clean_clip_end(end: float, words: list[dict]) -> tuple[float, float]:
    """
    Prevents mid-word cutoffs at the end of a clip!
    Returns (caption_end, video_end):
    - If a short concluding punchline/response immediately follows `end` (within 0.8s)
      and finishes within 4.5s (e.g., "Rushing! So you DO know the difference!"),
      extends both `caption_end` and `video_end` to include the full payoff!
    - Otherwise, ensures `video_end` stops cleanly BEFORE the next word begins.
    """
    # Find words that start near or after `end`
    subsequent_words = [
        w for w in words
        if float(w.get("start", 0.0)) >= (end - 0.15)
    ]

    if not subsequent_words:
        return end, end + TAIL_PADDING_SEC

    first_next = subsequent_words[0]
    gap_to_next = float(first_next["start"]) - end

    # Case 1: Plenty of silence (> 0.85s) after `end` -> safe to add breathing room
    if gap_to_next >= 0.85:
        safe_pad = min(TAIL_PADDING_SEC, gap_to_next - 0.12)
        return end, end + max(0.2, safe_pad)

    # Case 2: Immediate dialogue continues (< 0.85s gap).
    # Look ahead up to 5.0 seconds for a clean sentence boundary + pause (the payoff!)
    extended_end = None
    for i, w in enumerate(subsequent_words):
        w_end = float(w["end"])
        if (w_end - end) > 5.0:
            break

        is_punct = w["word"].endswith((".", "?", "!", "।"))
        next_gap = (
            float(subsequent_words[i + 1]["start"]) - w_end
            if i + 1 < len(subsequent_words)
            else 2.0
        )
        if is_punct and next_gap >= 0.45:
            extended_end = w_end
            # Keep extending if there's one more short punchline sentence within 4.5s
            continue

    if extended_end is not None:
        print(f"   -> Extended clip end by +{extended_end - end:.1f}s to complete final payoff sentence.")
        return extended_end, extended_end + 0.5

    # Case 3: Next dialogue is a long unrelated paragraph -> cut cleanly BEFORE first_next starts
    clean_cut = max(end, float(first_next["start"]) - 0.08)
    return end, clean_cut


def extract_clip_with_intelligence(
    video_path: str,
    output_path: str,
    start: float,
    end: float,
    words: list[dict]
) -> None:
    """
    Extracts a clip (with smart sentence-boundary payoff resolution), dynamically crops to 9:16,
    mutes severe audio slurs, and burns bold Reels-safe captions.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(output_path).parent / "temp"
    temp_dir.mkdir(exist_ok=True)

    basename = Path(output_path).stem
    temp_clip_16_9 = str(temp_dir / f"{basename}_16_9.mp4")

    caption_end, video_end = _resolve_clean_clip_end(end, words)
    duration = max(0.1, video_end - start)
    encoder_args = _get_video_encoder_args()

    print(f"   -> [1/4] Extracting accurate 16:9 clip ({duration:.1f}s)...")
    subprocess.run([
        "ffmpeg", "-nostdin", "-y",
        "-ss", f"{start:.3f}",
        "-i", video_path,
        "-t", f"{duration:.3f}",
        *encoder_args,
        "-c:a", "aac", "-b:a", "128k",
        temp_clip_16_9
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print(f"   -> [2/4] Building sanitized captions & checking audio moderation...")
    caption_events, mute_intervals = build_caption_events(words, start, caption_end)
    print(f"          {len(caption_events)} caption events created ({len(mute_intervals)} audio slur mute dips).")

    print(f"   -> [3/4] Tracking faces, side profiles & camera cuts...")
    smoothed_centers, width, height, fps = get_smooth_crop_centers(temp_clip_16_9, sample_fps=6.0)

    target_h = int(height) - (int(height) % 2)
    target_w = int(target_h * 9 / 16)
    target_w = target_w if target_w % 2 == 0 else target_w + 1

    audio_args = _build_audio_mute_filter(mute_intervals)

    print(f"   -> [4/4] Rendering final 9:16 video ({target_w}x{target_h}) via hardware encoder...")

    ffmpeg_cmd = [
        "ffmpeg", "-nostdin", "-y",
        "-loglevel", "error",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{target_w}x{target_h}",
        "-pix_fmt", "bgr24",
        "-r", str(fps),
        "-i", "-",              # stdin: cropped + captioned frames
        "-i", temp_clip_16_9,   # audio stream
        *encoder_args,
        *audio_args,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        "-movflags", "+faststart",
        output_path
    ]

    cap = cv2.VideoCapture(temp_clip_16_9)
    process = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE
    )

    event_idx = 0
    num_events = len(caption_events)

    try:
        frame_idx = 0
        for center in smoothed_centers:
            ret, frame = cap.read()
            if not ret:
                break

            x1 = int(round(center - target_w / 2.0))
            if x1 < 0:
                x1 = 0
            elif x1 + target_w > width:
                x1 = max(0, width - target_w)
            x2 = x1 + target_w

            cropped_frame = frame[:target_h, x1:x2].copy()

            t = frame_idx / fps
            while event_idx < num_events and caption_events[event_idx]["end"] <= t:
                event_idx += 1

            active_event = None
            if event_idx < num_events and caption_events[event_idx]["start"] <= t < caption_events[event_idx]["end"]:
                active_event = caption_events[event_idx]

            if active_event is not None:
                cropped_frame = render_caption_on_frame(cropped_frame, active_event)

            if process.stdin:
                process.stdin.write(cropped_frame.data)
            frame_idx += 1

    except BrokenPipeError:
        pass
    finally:
        try:
            if process.stdin and not process.stdin.closed:
                process.stdin.close()
        except Exception:
            pass

        stderr_bytes = b""
        try:
            if process.stderr:
                stderr_bytes = process.stderr.read()
        except Exception:
            pass

        process.wait()
        cap.release()

        if process.returncode != 0:
            err_msg = stderr_bytes.decode("utf-8", errors="replace")
            print(f"   ⚠ FFmpeg error: {err_msg[-400:]}")
        else:
            print(f"   ✓ Saved: {output_path}")

    try:
        os.remove(temp_clip_16_9)
    except OSError:
        pass


def extract_all_clips(
    video_path: str,
    clips: list[dict],
    output_dir: str,
    words: list[dict]
) -> None:
    """
    Extract multiple clips from a video using full video intelligence.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for index, clip in enumerate(clips):
        start = float(clip["start"])
        end = float(clip["end"])
        title = clip.get("title", f"clip_{index}")

        clean_title = "".join(c if c.isalnum() else "_" for c in title).strip("_")
        output_path = f"{output_dir}/{index:02d}_{clean_title}.mp4"

        print(f"\nProcessing Clip {index + 1}/{len(clips)}: {title} ({start:.1f}s -> {end:.1f}s)")
        extract_clip_with_intelligence(video_path, output_path, start, end, words)

    print(f"\n✓ All {len(clips)} clips processed.")


























# import os
# import platform
# import subprocess
# from pathlib import Path
# import numpy as np

# from app.services.face_tracking_service import get_smooth_crop_centers
# from app.services.caption_service import (
#     build_caption_events,
#     export_ass_captions
# )


# def _get_video_encoder_args() -> list[str]:
#     """
#     Uses Apple Silicon / macOS hardware acceleration (h264_videotoolbox)
#     if available to eliminate CPU heating during encoding; otherwise uses
#     thread-capped libx264 veryfast.
#     """
#     if platform.system() == "Darwin":
#         try:
#             res = subprocess.run(
#                 ["ffmpeg", "-hide_banner", "-encoders"],
#                 capture_output=True, text=True, check=True
#             )
#             if "h264_videotoolbox" in res.stdout:
#                 return ["-c:v", "h264_videotoolbox", "-b:v", "4500k", "-pix_fmt", "yuv420p"]
#         except Exception:
#             pass

#     return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-threads", "4", "-pix_fmt", "yuv420p"]


# def _build_ffmpeg_crop_x_expr(
#     smoothed_centers: list[float],
#     width: int,
#     target_w: int,
#     fps: float,
#     num_keyframes: int = 8
# ) -> str:
#     """
#     Converts the smoothed face center trajectory into a compact FFmpeg expression
#     so FFmpeg can dynamically pan the 9:16 crop window natively in C without Python loops!
#     """
#     max_x = max(0, width - target_w)
#     if not smoothed_centers:
#         return str(int(max_x // 2))

#     x_positions = [
#         min(max_x, max(0.0, c - target_w / 2.0))
#         for c in smoothed_centers
#     ]

#     # If speaker stays mostly stationary (< 25px drift), use a clean static crop
#     if (max(x_positions) - min(x_positions)) < 25.0:
#         return str(int(round(float(np.median(x_positions)))))

#     # Otherwise, sample `num_keyframes` points and build a smooth piecewise-linear FFmpeg expr
#     n = len(x_positions)
#     indices = np.linspace(0, n - 1, min(num_keyframes, n), dtype=int)
#     times = [round(idx / fps, 2) for idx in indices]
#     xs = [int(round(x_positions[idx])) for idx in indices]

#     expr = str(xs[-1])
#     for i in range(len(indices) - 2, -1, -1):
#         t0, t1 = times[i], times[i + 1]
#         x0, x1 = xs[i], xs[i + 1]
#         dt = max(0.01, round(t1 - t0, 2))
#         dx = x1 - x0
#         seg = f"{x0}" if dx == 0 else f"({x0}+{dx}*(t-{t0})/{dt})"
#         expr = f"if(lte(t,{t1}),{seg},{expr})"

#     return f"clip({expr},0,{max_x})"


# def _escape_ffmpeg_filter_path(path_str: str) -> str:
#     """Escapes file paths for FFmpeg's subtitles/ass filter."""
#     return (
#         path_str.replace("\\", "/")
#         .replace(":", "\\:")
#         .replace("'", "\\'")
#         .replace("[", "\\[")
#         .replace("]", "\\]")
#     )


# def extract_clip_with_intelligence(
#     video_path: str,
#     output_path: str,
#     start: float,
#     end: float,
#     words: list[dict]
# ) -> None:
#     """
#     Extracts a clip, dynamically crops to 9:16 using sampled face tracking,
#     and burns animated word-level subtitles using a single native FFmpeg filter pass.
#     """
#     Path(output_path).parent.mkdir(parents=True, exist_ok=True)
#     temp_dir = Path(output_path).parent / "temp"
#     temp_dir.mkdir(exist_ok=True)

#     basename = Path(output_path).stem
#     temp_clip_16_9 = str(temp_dir / f"{basename}_16_9.mp4")
#     temp_ass_path = str(temp_dir / f"{basename}.ass")

#     duration = max(0.1, end - start)
#     encoder_args = _get_video_encoder_args()

#     print(f"   -> [1/4] Extracting raw 16:9 clip ({duration:.1f}s)...")
#     # Fast input seek with lightweight encode so start timestamp is frame-accurate (no caption drift)
#     subprocess.run([
#         "ffmpeg", "-nostdin", "-y",
#         "-ss", f"{start:.3f}",
#         "-i", video_path,
#         "-t", f"{duration:.3f}",
#         *encoder_args,
#         "-c:a", "aac", "-b:a", "128k",
#         temp_clip_16_9
#     ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

#     print(f"   -> [2/4] Building .ass caption file...")
#     caption_events = build_caption_events(words, start, end)
#     print(f"          {len(caption_events)} caption events created.")

#     print(f"   -> [3/4] Sampling faces (2 FPS) for dynamic crop...")
#     smoothed_centers, width, height, fps = get_smooth_crop_centers(temp_clip_16_9, sample_fps=2.0)

#     target_h = int(height) - (int(height) % 2)
#     target_w = int(target_h * 9 / 16)
#     target_w = target_w if target_w % 2 == 0 else target_w + 1

#     export_ass_captions(caption_events, temp_ass_path, play_res_x=target_w, play_res_y=target_h)
#     crop_x_expr = _build_ffmpeg_crop_x_expr(smoothed_centers, width, target_w, fps)
#     escaped_ass = _escape_ffmpeg_filter_path(temp_ass_path)

#     print(f"   -> [4/4] Rendering final 9:16 video ({target_w}x{target_h}) via native FFmpeg...")
#     vf_chain = f"crop=w={target_w}:h={target_h}:x='{crop_x_expr}':y=0,ass='{escaped_ass}'"

#     ffmpeg_cmd = [
#         "ffmpeg", "-nostdin", "-y",
#         "-i", temp_clip_16_9,
#         "-vf", vf_chain,
#         *encoder_args,
#         "-c:a", "copy",
#         "-movflags", "+faststart",
#         output_path
#     ]

#     res = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
#     if res.returncode != 0:
#         print("   ⚠ FFmpeg `ass` filter unavailable, rendering crop natively + captions...")
#         # Fallback if Homebrew FFmpeg was compiled without libass:
#         vf_crop_only = f"crop=w={target_w}:h={target_h}:x='{crop_x_expr}':y=0"
#         subprocess.run([
#             "ffmpeg", "-nostdin", "-y",
#             "-i", temp_clip_16_9,
#             "-vf", vf_crop_only,
#             *encoder_args,
#             "-c:a", "copy",
#             output_path
#         ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

#     print(f"   ✓ Saved: {output_path}")

#     for temp_file in (temp_clip_16_9, temp_ass_path):
#         try:
#             os.remove(temp_file)
#         except OSError:
#             pass


# def extract_all_clips(
#     video_path: str,
#     clips: list[dict],
#     output_dir: str,
#     words: list[dict]
# ) -> None:
#     """
#     Extract multiple clips from a video using full video intelligence.
#     """
#     Path(output_dir).mkdir(parents=True, exist_ok=True)

#     for index, clip in enumerate(clips):
#         start = float(clip["start"])
#         end = float(clip["end"])
#         title = clip.get("title", f"clip_{index}")

#         clean_title = "".join(c if c.isalnum() else "_" for c in title).strip("_")
#         output_path = f"{output_dir}/{index:02d}_{clean_title}.mp4"

#         print(f"\nProcessing Clip {index + 1}/{len(clips)}: {title} ({start:.1f}s -> {end:.1f}s)")
#         extract_clip_with_intelligence(video_path, output_path, start, end, words)

#     print(f"\n✓ All {len(clips)} clips processed.")





 
   