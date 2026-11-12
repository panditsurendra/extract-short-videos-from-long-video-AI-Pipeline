[← Back to Main Menu](../../README.md)


# 🚧 Engineering Challenges & Architectural Lessons

Building the AI-Powered Long-to-Shorts Video Pipeline involved solving several heavy media processing, computer vision, and AI orchestration bottlenecks. This document outlines the core technical challenges faced during development and the engineering trade-offs made to resolve them.

---

## 1. Hardware Overheating & Thermal Throttling
**The Problem:**
Early iterations of the pipeline relied entirely on heavy, sequential CPU operations. Running local Whisper transcription models alongside 30 FPS OpenCV frame-by-frame processing (`cv2.VideoWriter`) caused severe CPU spikes. For a 30-minute podcast, laptops would quickly overheat, trigger thermal throttling, and push processing times well past the 20-minute mark.

**The Solution:**
I completely re-architected the pipeline to minimize local CPU footprint:
* **Cloud API Offloading:** Shifted transcription to ultra-fast cloud inference APIs (Groq Whisper) and offloaded clip-selection semantic processing to Gemini/Groq LLMs.
* **Hardware-Accelerated Encoding:** Replaced OpenCV's inefficient `cv2.VideoWriter` with Apple Silicon's native hardware encoder (`h264_videotoolbox`) and `libx264` fallbacks via FFmpeg.
* **Zero-Copy Rendering Pipeline:** Instead of applying `.putText` to thousands of individual frames in Python, the pipeline now calculates text bounding boxes in memory, renders unique subtitle events using Pillow just *once*, and pipes raw byte data directly into the FFmpeg hardware encoder via `stdin`. Processing time dropped to mere seconds without triggering laptop fans.

---

## 2. Computer Vision: Face Tracking vs. Scene Cuts
**The Problem:**
Standard facial recognition (`haarcascade_frontalface_default.xml`) works well for solo creators looking into the lens, but podcasts and movies (e.g., *Whiplash*) utilize multiple camera angles, side profiles, and rapid shot-reverse-shot cuts. A basic Exponential Moving Average (EMA) crop tracker would "go blind" when a subject turned sideways, or it would slowly, awkwardly pan across the screen during a hard camera cut.

**The Solution:**
Built a scene-cut aware, multi-axis tracking module:
* **Multi-Angle Detection:** Integrated MediaPipe (which handles head tilts/profiles better) and layered it with left/right Haar cascade fallbacks. If the subject's face isn't visible (e.g., an over-the-shoulder shot), it falls back to a visual edge/salience detector to center on the action.
* **Histogram Scene-Cut Detection:** Added a `cv2.calcHist` Bhattacharyya distance check. When a hard camera cut is detected, the 9:16 crop window snaps *instantly* to the new speaker's coordinates rather than smoothing across empty space, mimicking a human video editor's precision.

---

## 3. Algorithmic Suppression (Platform Safety)
**The Problem:**
Instagram Reels, TikTok, and YouTube Shorts employ strict automated speech-to-text moderation. Uncensored profanity or severe slurs present in raw podcast audio frequently trigger algorithmic shadowbans, severely restricting content reach. 

**The Solution:**
Programmatic media censorship without losing A/V synchronization:
* Built a text-sanitization engine that catches phonetic errors and masks profanity (e.g., `sh*t`, `f*ck`) in the generated captions.
* For severe slurs, the pipeline captures the exact millisecond start/end timestamps from the Whisper JSON output and dynamically constructs an FFmpeg audio filter (`-af volume=enable='between(t,start,end)':volume=0`) to dip the audio precisely over the slur.

---

## 4. Subtitle Typography & Payoff Cutoffs
**The Problem:**
Default programmatic subtitles look pixelated, clash with mobile UI elements, and ruin the pacing of the video. Furthermore, blindly cropping a video at an LLM's suggested `end` timestamp often sliced words in half or cut off the punchline of a joke.

**The Solution:**
* **Typography:** Ditched OpenCV's thin `FONT_HERSHEY` fonts for `PIL.ImageDraw`, enabling anti-aliased TrueType fonts (like Impact/Arial Bold) with thick black stroke outlines. Positioned the captions exactly at 68% height to float safely above Instagram's UI.
* **Smart Pacing:** Added a `_resolve_clean_clip_end()` algorithm. If the LLM selects a clip that ends right before a concluding sentence or punchline, the pipeline scans ahead in the transcript. If a natural sentence boundary exists within the next 5 seconds, it automatically extends the video and captions to include the full payoff.

---

## 5. Streaming Large Media via ASGI / Starlette
**The Problem:**
When rendering a Streamlit web UI to preview the final clips, the underlying Starlette ASGI server crashed with a `RuntimeError: Expected ASGI message 'http.response.start', but got 'http.response.body'`. This happens because modern browsers use HTTP `206 Partial Content` range requests to stream `.mp4` files, which conflicts with Starlette's GZip middleware compression logic.

**The Solution:**
To bypass the broken ASGI media endpoint without downgrading the entire Streamlit application, I implemented an inline Base64 HTML5 video renderer. By injecting `<video src="data:video/mp4;base64,...">` directly into the DOM, the browser handles playback purely on the client side, eliminating server-side chunking crashes and delivering instant playback.

---

## 6. LLM Rate Limits & Concurrency (Groq TPM)
**The Problem:**
Passing large transcript chunks into LLMs for semantic clip selection frequently hit Groq's 8,000 Tokens Per Minute (TPM) rate limits, causing API `429` errors and halting the pipeline.

**The Solution:**
Implemented a robust multi-model fallback and rotation architecture in `llm_service.py`:
* The system actively queries the active models on the API key and rotates through them sequentially (e.g., `llama-3.3-70b` -> `gpt-oss-20b`).
* If a 429 rate limit is hit, regex parsing intercepts Groq's "Please try again in X.XXs" message, executing a precise `time.sleep()` before retrying.
* Integrated Gemini 3.8 Flash as an automatic fallback, which offers massive 1M+ token context windows for processing entire podcast transcripts in a single shot without strict TPM bottlenecks.