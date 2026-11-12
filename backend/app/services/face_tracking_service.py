
import os
import cv2
import numpy as np


def _detect_face_center_x(
    small_bgr: np.ndarray,
    small_gray: np.ndarray,
    mp_detector,
    frontal_cascade,
    profile_cascade,
    scale: float,
    width: int
) -> float | None:
    """
    Detects the primary subject's X center across the full frame (leftmost to rightmost):
    1. MediaPipe Face Detection (if available — handles profiles & angles best)
    2. Frontal Haar Cascade
    3. Left & Right Profile Haar Cascades
    4. High-contrast subject salience fallback (for extreme close-ups / over-the-shoulder shots)
    """
    h_s, w_s = small_gray.shape[:2]

    # 1. Try MediaPipe first if available
    if mp_detector is not None:
        rgb = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)
        results = mp_detector.process(rgb)
        if results.detections:
            best_det = max(
                results.detections,
                key=lambda d: (
                    d.location_data.relative_bounding_box.width
                    * d.location_data.relative_bounding_box.height
                )
            )
            bbox = best_det.location_data.relative_bounding_box
            center_x = (bbox.xmin + bbox.width / 2.0) * width
            return float(np.clip(center_x, 0, width))

    candidates = []

    # 2. Frontal faces
    frontals = frontal_cascade.detectMultiScale(
        small_gray, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20)
    )
    for (x, y, w, h) in frontals:
        candidates.append((w * h * 1.2, (x + w / 2.0) / scale))

    # 3. Profile faces (left-facing and right-facing via horizontal flip)
    if not candidates and not profile_cascade.empty():
        profiles_left = profile_cascade.detectMultiScale(
            small_gray, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20)
        )
        for (x, y, w, h) in profiles_left:
            candidates.append((w * h, (x + w / 2.0) / scale))

        flipped = cv2.flip(small_gray, 1)
        profiles_right = profile_cascade.detectMultiScale(
            flipped, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20)
        )
        for (x, y, w, h) in profiles_right:
            orig_x = w_s - (x + w)
            candidates.append((w * h, (orig_x + w / 2.0) / scale))

    if candidates:
        # Pick the most prominent face anywhere on screen (left, center, or right)
        best_center = max(candidates, key=lambda item: item[0])[1]
        return float(np.clip(best_center, 0, width))

    # 4. Fallback for extreme close-ups (e.g. half-face on edge of screen or dark background):
    # Find the horizontal center of mass of texture/lighting so we never stare at black emptiness
    sobel_x = cv2.Sobel(small_gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(small_gray, cv2.CV_32F, 0, 1, ksize=3)
    energy = np.abs(sobel_x) + np.abs(sobel_y) + (small_gray.astype(np.float32) * 0.5)

    col_weights = energy.mean(axis=0)
    if col_weights.max() > 15.0:
        # Find the brightest/sharpest 9:16-width window across the horizontal axis
        win_w = max(1, int(round((h_s * 9.0 / 16.0))))
        if win_w < w_s:
            kernel = np.ones(win_w, dtype=np.float32)
            window_scores = np.convolve(col_weights, kernel, mode="valid")
            best_start = int(np.argmax(window_scores))
            best_center_small = best_start + win_w / 2.0
            return float(np.clip(best_center_small / scale, 0, width))

    return None


def get_smooth_crop_centers(video_path: str, sample_fps: float = 6.0):
    """
    Scene-cut aware face & subject tracker.
    - Detects hard camera cuts so the crop window snaps immediately when the camera switches speakers.
    - Detects frontal faces, side-profile faces (left & right), and visual salience at the screen edges.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video stream or file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    stride = max(1, int(round(fps / sample_fps)))

    # Try initializing MediaPipe Face Detection if installed
    mp_detector = None
    try:
        # pyrefly: ignore [missing-import]
        import mediapipe as mp
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_detection"):
            mp_detector = mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.45
            )
    except Exception:
        mp_detector = None

    frontal_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    profile_path = os.path.join(cv2.data.haarcascades, "haarcascade_profileface.xml")
    frontal_cascade = cv2.CascadeClassifier(frontal_path)
    profile_cascade = cv2.CascadeClassifier(profile_path)

    detect_width = 360
    scale = detect_width / float(width) if width > detect_width else 1.0

    # Store per-frame raw centers and scene cut flags
    raw_centers = np.full(max(1, total_frames), width / 2.0, dtype=np.float32)
    scene_cuts = [0]

    prev_hist = None
    last_center = width / 2.0
    last_sample_frame = 0

    frame_idx = 0
    while True:
        if frame_idx % stride != 0:
            if not cap.grab():
                break
            if frame_idx < len(raw_centers):
                raw_centers[frame_idx] = last_center
            frame_idx += 1
            continue

        ret, frame = cap.read()
        if not ret:
            break

        small_bgr = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        small_gray = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2GRAY)

        # 1. Check if a camera cut (scene change) occurred since the last sample
        hist = cv2.calcHist([small_gray], [0], None, [32], [0, 256])
        cv2.normalize(hist, hist)
        if prev_hist is not None:
            hist_diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
            if hist_diff > 0.32:
                scene_cuts.append(frame_idx)
        prev_hist = hist

        # 2. Detect face/subject center anywhere from far-left to far-right
        detected_x = _detect_face_center_x(
            small_bgr, small_gray, mp_detector,
            frontal_cascade, profile_cascade, scale, width
        )
        if detected_x is not None:
            last_center = detected_x

        # Fill from last sample to current frame
        end_fill = min(frame_idx + 1, len(raw_centers))
        raw_centers[last_sample_frame:end_fill] = last_center
        last_sample_frame = frame_idx + 1
        frame_idx += 1

    cap.release()
    if mp_detector is not None:
        mp_detector.close()

    actual_frames = max(1, min(frame_idx, len(raw_centers)))
    raw_centers = raw_centers[:actual_frames]
    scene_cuts.append(actual_frames)

    # 3. Smooth *independently* inside each camera shot so scene cuts snap instantly!
    smoothed_centers = np.copy(raw_centers)
    for i in range(len(scene_cuts) - 1):
        s_start = scene_cuts[i]
        s_end = scene_cuts[i + 1]
        if s_end <= s_start:
            continue

        shot_vals = raw_centers[s_start:s_end]
        # If the speaker in this shot doesn't walk across the room, lock onto their median position
        if ( np.max(shot_vals) - np.min(shot_vals) ) < (width * 0.12):
            smoothed_centers[s_start:s_end] = float(np.median(shot_vals))
        else:
            # Smoothly pan within the shot using zero-lag forward-backward EMA
            alpha = 0.15
            fwd = np.zeros_like(shot_vals)
            val = shot_vals[0]
            for idx, c in enumerate(shot_vals):
                val = alpha * c + (1.0 - alpha) * val
                fwd[idx] = val

            bwd = np.zeros_like(fwd)
            val = fwd[-1]
            for idx in range(len(fwd) - 1, -1, -1):
                val = alpha * fwd[idx] + (1.0 - alpha) * val
                bwd[idx] = val
            smoothed_centers[s_start:s_end] = bwd

    return smoothed_centers.tolist(), width, height, fps












# import os
# import cv2
# import numpy as np


# def get_smooth_crop_centers(video_path: str, sample_fps: float = 2.0):
#     """
#     Analyzes a clip by sampling only `sample_fps` frames per second (using cap.grab()
#     to skip decoding intermediate frames), cutting CPU usage by ~93%.
#     Returns (smoothed_centers, width, height, fps).
#     """
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         raise ValueError(f"Error opening video stream or file: {video_path}")

#     fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#     stride = max(1, int(round(fps / sample_fps)))

#     cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
#     face_cascade = cv2.CascadeClassifier(cascade_path)

#     sampled_indices = []
#     sampled_centers = []
#     last_known_center = width / 2.0

#     # Downscale target width of 320px for ultra-fast detection
#     detect_width = 320
#     scale = detect_width / float(width) if width > detect_width else 1.0

#     frame_idx = 0
#     while True:
#         if frame_idx % stride != 0:
#             # Skip frame without decoding pixels!
#             if not cap.grab():
#                 break
#             frame_idx += 1
#             continue

#         ret, frame = cap.read()
#         if not ret:
#             break

#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         small_gray = cv2.resize(gray, (0, 0), fx=scale, fy=scale)

#         faces = face_cascade.detectMultiScale(
#             small_gray,
#             scaleFactor=1.15,
#             minNeighbors=5,
#             minSize=(24, 24)
#         )

#         if len(faces) > 0:
#             # Pick largest face
#             largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
#             x, y, w, h = largest_face
#             best_center = (x + w / 2.0) / scale
#             last_known_center = best_center

#         sampled_indices.append(frame_idx)
#         sampled_centers.append(last_known_center)
#         frame_idx += 1

#     cap.release()

#     actual_frames = max(total_frames, frame_idx, 1)
#     if not sampled_centers:
#         return [width / 2.0] * actual_frames, width, height, fps

#     # Interpolate sampled points across all frames
#     all_frames_idx = np.arange(actual_frames)
#     interp_centers = np.interp(all_frames_idx, sampled_indices, sampled_centers)

#     # Forward-backward EMA smoothing to eliminate camera lag & jitter
#     alpha = 0.06
#     forward = np.zeros_like(interp_centers)
#     val = interp_centers[0]
#     for i, c in enumerate(interp_centers):
#         val = alpha * c + (1.0 - alpha) * val
#         forward[i] = val

#     backward = np.zeros_like(forward)
#     val = forward[-1]
#     for i in range(len(forward) - 1, -1, -1):
#         val = alpha * forward[i] + (1.0 - alpha) * val
#         backward[i] = val

#     return backward.tolist(), width, height, fps




# import cv2
# import os


# def get_smooth_crop_centers(video_path: str):
#     """
#     Analyzes a short clip, detects faces using OpenCV Haar Cascades,
#     and returns a smoothed list of target X coordinates (centers)
#     to create a "virtual cameraman" effect.
#     """
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         raise ValueError(f"Error opening video stream or file: {video_path}")
        
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
#     # Load OpenCV pre-trained Face Detector
#     cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
#     face_cascade = cv2.CascadeClassifier(cascade_path)
    
#     target_centers = []
#     last_known_center = width / 2
    
#     # 1. Face Detection pass
#     for _ in range(total_frames):
#         ret, frame = cap.read()
#         if not ret:
#             break
            
#         # Convert to grayscale for Haar Cascade
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
#         # Scale down for significantly faster detection
#         scale = 0.5
#         small_gray = cv2.resize(gray, (0, 0), fx=scale, fy=scale)
        
#         faces = face_cascade.detectMultiScale(
#             small_gray, 
#             scaleFactor=1.1, 
#             minNeighbors=5, 
#             minSize=(30, 30)
#         )
        
#         if len(faces) > 0:
#             largest_area = 0
#             best_center = last_known_center
            
#             for (x, y, w, h) in faces:
#                 area = w * h
#                 if area > largest_area:
#                     largest_area = area
#                     # Calculate center and scale back to original resolution
#                     best_center = (x + w / 2) / scale
                    
#             target_centers.append(best_center)
#             last_known_center = best_center
#         else:
#             target_centers.append(last_known_center)
            
#     cap.release()
    
#     # 2. Smooth trajectory using Exponential Moving Average (EMA) to prevent camera shake
#     smoothed_centers = []
#     alpha = 0.05 # Lower value = smoother tracking
    
#     current_val = target_centers[0] if target_centers else width / 2
#     for center in target_centers:
#         current_val = (alpha * center) + ((1 - alpha) * current_val)
#         smoothed_centers.append(current_val)
        
#     return smoothed_centers, width, height, fps
