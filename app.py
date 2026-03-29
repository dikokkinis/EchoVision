# app.py
import gradio as gr
import tempfile, os
from src.extractor import extract_frames_and_audio
from src.encoders import CLAPEncoder, CLIPPatchEncoder
from src.localizer import SoundLocalizer
from src.visualizer import overlay_heatmap
import cv2

audio_enc = CLAPEncoder()
clip_enc = CLIPPatchEncoder()
localizer = SoundLocalizer(audio_enc, clip_enc)

def process_video(video_path):
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
        out_path = "/tmp/echovision_output.mp4"
        h, w = out_frames[0].shape[:2]
        writer = cv2.VideoWriter(
            out_path, cv2.VideoWriter_fourcc(*"mp4v"), 2, (w, h)
        )
        for f in out_frames:
            writer.write(f)
        writer.release()
        return out_path

demo = gr.Interface(
    fn=process_video,
    inputs=gr.Video(label="Upload a video with clear sound sources"),
    outputs=gr.Video(label="Sound source localization"),
    title="EchoVision — Zero-Shot Audio-Visual Sound Localization",
    description="Highlights regions of the video frame that correspond "
                "to the audio using CLAP + CLIP cross-modal attention. "
                "No training required.",
    examples=[["assets/examples/dog_barking.mp4"],
              ["assets/examples/guitar_playing.mp4"]]
)

if __name__ == "__main__":
    demo.launch()