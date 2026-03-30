"""This script acts as a test for the extractor module.
It loads a sample youtube video file and checks if the frames and audio are extracted correctly.
Simply run python test_extractor.py to execute the test.
"""
import sys, os
# Add the parent directory of 'src' to the search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.extractor import extract_frames_and_audio
import requests
import cv2

def download_video(url: str, out_path: str):
    print(f"Downloading video from {url}...")

    if os.path.exists(out_path):
        print(f"File {out_path} already exists. Skipping download.")
        return
    
    with requests.get(url, stream=True) as response:
        response.raise_for_status()

        with open(out_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    print(f"Video saved successfully to {out_path}.")

def main():
    output_dir = "test_output"
    video_url = "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4"
    download_video(video_url, "sample_video.mp4")
    frames, timestamps, audio_path = extract_frames_and_audio("sample_video.mp4", output_dir, fps=1)

    assert len(frames) > 0, "No frames were extracted."
    #Save frames
    for i, frame in enumerate(frames):
        cv2.imwrite(f"{output_dir}/frames/frame_{i}.jpg", frame)
    
    print(f"Extracted {len(frames)} frames and audio saved at {audio_path}.")  
    #print(f"Timestamps of extracted frames: {timestamps}")
if __name__ == "__main__":
    main()