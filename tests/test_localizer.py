"""
This script acts as a test for the localizer module.
Simply replace the extracted audio path and the corresponding frame and run python test_localizer.py to execute the test.
"""
import sys, os
# Add the parent directory of 'src' to the search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
from src import localizer
from src.localizer import SoundLocalizer
from src.encoders import CLAPEncoder, CLIPPatchEncoder

def main():
    audio_enc = CLAPEncoder()
    clip_enc = CLIPPatchEncoder()  
    localizer = SoundLocalizer(audio_enc, clip_enc)

    audio_path = "test_output/audio.wav"
    frame_bgr = "test_output/frames/frame_12.jpg"  
    ts = 12.0  # Timestamp in seconds

    frame = cv2.imread(frame_bgr)
    sim_map = localizer.localize(audio_path, frame, ts)
    print("Similarity map shape:", sim_map.shape)

if __name__ == "__main__":
    main()