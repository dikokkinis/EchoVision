"""
Script to run the EchoVision process on a single video file.

Usage:
    python run_simple.py --video_path path/to/your/video.mp4 or url --output_dir path/to/output/directory
"""
import argparse
import tempfile, os
import cv2
from src.localizer import SoundLocalizer
from src.extractor import extract_frames_and_audio
from src.encoders import CLAPEncoder, CLIPPatchEncoder
from src.visualizer import overlay_heatmap
from utils import download_video

def process_video(video_path, localizer, folder, fps=2):
    with tempfile.TemporaryDirectory() as tmp:
        frames, timestamps, audio_path = extract_frames_and_audio(
            video_path, tmp, fps=fps
        )
        out_frames = []
        for frame, ts in zip(frames, timestamps):
            sim_map = localizer.localize(audio_path, frame, ts)
            annotated = overlay_heatmap(frame, sim_map)
            out_frames.append(annotated)

        # Write output video
        output_dir = os.mkdir(folder) if not os.path.exists(folder) else folder
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
    argparser.add_argument("--fps", type=int, required=False, help="Frames per second for video processing")
    args = argparser.parse_args()

    #Check if output_dir exists, if not create it
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    #Check if video_path is a URL and download if necessary
    video_file_path = " "
    if args.video_path.startswith("http://") or args.video_path.startswith("https://"):
        print(f"Downloading video from URL: {args.video_path}")
        video_file_path = os.path.join(args.output_dir, "downloaded_video.mp4")
        download_video(args.video_path, video_file_path)
        print(f"Video downloaded to: {video_file_path}")
    elif os.path.exists(args.video_path):
        print(f"Using local video file: {args.video_path}")
        video_file_path = args.video_path
    else:
        raise ValueError(f"Invalid video path: {args.video_path}")

    audio_enc = CLAPEncoder()
    clip_enc = CLIPPatchEncoder()
    localizer = SoundLocalizer(audio_enc, clip_enc)

    print(f"Processing video with {args.fps} FPS...")
    output_video = process_video(video_file_path, localizer, args.output_dir, args.fps)
    print(f"Processed video saved to: {output_video}")

if __name__ == "__main__":
    main()
