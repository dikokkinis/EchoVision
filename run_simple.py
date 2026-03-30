"""
Script to run the EchoVision process on a single video file.

Usage:
    python run_simple.py --video_path path/to/your/video.mp4 --output_dir path/to/output/directory
"""
import argparse
import tempfile, os
import cv2
from src.localizer import SoundLocalizer
from src.extractor import extract_frames_and_audio
from src.encoders import CLAPEncoder, CLIPPatchEncoder
from src.visualizer import overlay_heatmap

def process_video(video_path, localizer, output_dir):
    with tempfile.TemporaryDirectory() as tmp:
        frames, timestamps, audio_path = extract_frames_and_audio(
            video_path, tmp, fps=2
        )
        out_frames = []
        for frame, ts in zip(frames, timestamps):
            sim_map = localizer.localize(audio_path, frame, ts)
            annotated = overlay_heatmap(frame, sim_map)
            out_frames.append(annotated)

        # Write output video
        output_dir = os.mkdir(output_dir) if not os.path.exists(output_dir) else output_dir
        out_path = os.path.join(output_dir, "echovision_output.mp4")
        h, w = out_frames[0].shape[:2]
        writer = cv2.VideoWriter(
            out_path, cv2.VideoWriter_fourcc(*"mp4v"), 2, (w, h)
        )
        for f in out_frames:
            writer.write(f)
        writer.release()
        return out_path
    
def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--video_path", type=str, required=True, help="Path to the input video file")
    argparser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory")
    args = argparser.parse_args()

    audio_enc = CLAPEncoder()
    clip_enc = CLIPPatchEncoder()
    localizer = SoundLocalizer(audio_enc, clip_enc)

    output_video = process_video(args.video_path, localizer, args.output_dir)
    print(f"Processed video saved to: {output_video}")

if __name__ == "__main__":
    main()
