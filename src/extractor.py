import subprocess, os
import cv2
import numpy as np

def extract_frames_and_audio(video_path: str, out_dir: str, fps: int = 1):
    """This is the main function to extract audio and frames from a video file."""
    os.makedirs(f"{out_dir}/frames", exist_ok=True)
    audio_path = os.path.join(out_dir, "audio.wav")
    # Extract audio
    subprocess.run([
        "ffmpeg", "-i", video_path, "-ac", "1", "-ar", "48000",
        audio_path, "-y"
    ], capture_output=True, check=True)
    # Extract frames at target fps
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)
    frames, timestamps = [], []
    idx = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        if idx % frame_interval == 0:
            frames.append(frame)
            timestamps.append(idx / video_fps)
        idx += 1
    cap.release()
    return frames, timestamps, audio_path