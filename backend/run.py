import os
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


VIDEO_PATH = "../assets/input/input_video.mp4"
AUDIO_PATH = "../assets/audio/audio.wav"
TRANSCRIPT_PATH = "../assets/transcripts/aligned_transcript.json"
CLIPS_DIR = "../assets/clips"
COVER_PATH = "../assets/input/sc3.png"


def main():
    print("\n=== Long to Shorts Pipeline ===")
    
    # 1. Audio Extraction
    if not os.path.exists(AUDIO_PATH):
        print("\n1. Extracting audio...")
        extract_audio(VIDEO_PATH, AUDIO_PATH)
    else:
        print("\n1. Audio already extracted. Skipping.")

    # 2. Transcription and Alignment
    if not os.path.exists(TRANSCRIPT_PATH):
        print("\n2. Transcribing and aligning...")
        transcribe_and_align(AUDIO_PATH, TRANSCRIPT_PATH)
    else:
        print("\n2. Transcript already exists. Skipping.")

    # 3. Processing Transcript
    print("\n3. Processing transcript...")
    transcript = load_transcript(TRANSCRIPT_PATH)
    words = extract_words(transcript)
    sentences = build_sentences(words)
    chunks = create_chunks(sentences)
    print(f"   Created {len(chunks)} candidate chunks.")

    # 4. Clip Selection
    print("\n4. Selecting best clips using LLM...")
    clips = select_clips(chunks)
    print(f"   LLM selected {len(clips)} clips.")
    
    if not clips:
        print("   No clips selected. Exiting.")
        return

    # 5. Extracting Clips with Video Intelligence
    print("\n5. Extracting clips and rendering 9:16 with dynamic subtitles...")
    extract_all_clips(VIDEO_PATH, clips, CLIPS_DIR, words)
    ##### Pass your cover image path here (e.g. "../data/input/cover.jpg")
    # extract_all_clips(
    #     video_path=VIDEO_PATH,
    #     clips=clips,
    #     output_dir=CLIPS_DIR,
    #     words=words,
    #     cover_image_path=COVER_PATH  # Optional cover photo
    # )
    
    print("\n=== Pipeline Complete ===")
    print(f"Clips saved to: {os.path.abspath(CLIPS_DIR)}")


if __name__ == "__main__":
    main()