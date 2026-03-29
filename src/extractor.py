# src/extractor.py
import subprocess, os
import cv2
import numpy as np

def extract_frames_and_audio(video_path: str, out_dir: str, fps: int = 1):
    """This is the main function to extract audio and frames from a video file."""
    os.makedirs(f"{out_dir}/frames", exist_ok=True)
    # Extract audio
    subprocess.run([
        "ffmpeg", "-i", video_path, "-ac", "1", "-ar", "22050",
        f"{out_dir}/audio.wav", "-y"
    ], check=True)
    # Extract frames at target fps
    cap = cv2.VideoCapture(video_path)
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
    return frames, timestamps, f"{out_dir}/audio.wav"