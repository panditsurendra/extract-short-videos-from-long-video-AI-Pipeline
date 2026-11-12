import os
import sys
from pathlib import Path

# Add backend/ to Python path so we can import services cleanly from the root directory
BACKEND_DIR = Path(__file__).parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

import streamlit as st
import json

from app.services.audio_service import extract_audio
from app.services.transcription_service import transcribe_and_align
from app.services.transcript_service import (
    load_transcript,
    extract_words,
    build_sentences
)
from app.services.chunk_service import create_chunks
from app.services.clip_selection_service import select_clips
from app.services.video_service import extract_all_clips

# Page Configuration
st.set_page_config(
    page_title="AI Long-to-Shorts Converter",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 AI Long-to-Shorts Video Pipeline")
st.markdown("Automate converting long-form videos into full-screen or vertical clips complete with word-level highlighted subtitles, audio safety moderation, and thumbnail cover integration.")

# ── Sidebar Configuration ─────────────────────────────────────────────
st.sidebar.header("📁 Pipeline Configuration")

input_dir = Path("assets/input")
input_dir.mkdir(parents=True, exist_ok=True)

existing_videos = [f.name for f in input_dir.glob("*.mp4")]
selected_video = st.sidebar.selectbox("Select Existing Video", options=["-- Upload New Video --"] + existing_videos)

video_path = None
if selected_video == "-- Upload New Video --":
    uploaded_file = st.sidebar.file_uploader("Upload MP4/MOV Video", type=["mp4", "mkv", "mov"])
    if uploaded_file:
        video_path = str(input_dir / uploaded_file.name)
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.sidebar.success(f"Saved: {uploaded_file.name}")
else:
    video_path = str(input_dir / selected_video)

cover_path_setting = st.sidebar.text_input("Cover Photo Path (Optional)", value="assets/input/sc3.png")
clips_dir = st.sidebar.text_input("Output Clips Directory", value="assets/clips")

audio_path = "assets/audio/audio.wav"
transcript_path = "assets/transcripts/aligned_transcript.json"

# Run Pipeline Button
if st.sidebar.button("🚀 Run Pipeline", type="primary"):
    if not video_path or not os.path.exists(video_path):
        st.error("Please select or upload a valid video file first!")
    else:
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # Stage 1: Audio Extraction
            status_text.text("Stage 1/5: Extracting Audio & Normalizing...")
            progress_bar.progress(10)
            if not os.path.exists(audio_path):
                extract_audio(video_path, audio_path)

            # Stage 2: Transcription & Alignment
            status_text.text("Stage 2/5: Transcribing & Aligning (Groq / Whisper)...")
            progress_bar.progress(30)
            if not os.path.exists(transcript_path) or os.path.getsize(transcript_path) == 0:
                transcribe_and_align(audio_path, transcript_path)

            # Stage 3: Transcript Processing
            status_text.text("Stage 3/5: Processing Transcript & Chunking...")
            progress_bar.progress(50)
            transcript = load_transcript(transcript_path)
            words = extract_words(transcript)
            sentences = build_sentences(words)
            chunks = create_chunks(sentences)

            # Stage 4: LLM Clip Selection
            status_text.text("Stage 4/5: Selecting Best Viral Clips via AI...")
            progress_bar.progress(70)
            clips = select_clips(chunks)

            if not clips:
                st.warning("No clips met the quality threshold.")
            else:
                # Stage 5: Video Rendering & Subtitle Burning
                status_text.text("Stage 5/5: Rendering Clips with Subtitles & Cover Image...")
                progress_bar.progress(90)

                cover_arg = cover_path_setting if os.path.exists(cover_path_setting) else None
                extract_all_clips(
                    video_path=video_path,
                    clips=clips,
                    output_dir=clips_dir,
                    words=words,
                    # cover_image_path=cover_arg
                )

                progress_bar.progress(100)
                status_text.success("✨ Pipeline Complete! All clips rendered successfully.")

        except Exception as e:
            st.error(f"Pipeline error: {str(e)}")


# ── Main Area: Display Generated Clips ────────────────────────────────
st.header("🎞️ Rendered Short Clips Preview")
clips_path = Path(clips_dir)

if clips_path.exists():
    clip_files = sorted(list(clips_path.glob("*.mp4")))
    if clip_files:
        # Create 4 columns instead of 2
        cols = st.columns(4)
        for idx, clip_file in enumerate(clip_files):
            # Route each video into one of the 4 columns
            with cols[idx % 4]:
                st.subheader(clip_file.stem.replace("_", " "))
                
                video_bytes = clip_file.read_bytes()
                st.video(video_bytes, format="video/mp4")
                st.download_button(
                    label=f"📥 Download",
                    data=video_bytes,
                    file_name=clip_file.name,
                    mime="video/mp4",
                    key=f"download_{idx}"
                )
                st.divider()
    else:
        st.info("No rendered clips found yet. Select your video and click **Run Pipeline** in the sidebar.")
else:
    st.info("Output clips directory does not exist yet.")