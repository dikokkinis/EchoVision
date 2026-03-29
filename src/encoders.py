# src/encoders.py
import torch
import numpy as np
from transformers import ClapModel, ClapProcessor
from PIL import Image
import librosa
import clip  # pip install git+https://github.com/openai/CLIP.git

class CLAPEncoder:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = ClapModel.from_pretrained(
            "laion/clap-htsat-unfused"
        ).to(self.device).eval()
        self.processor = ClapProcessor.from_pretrained(
            "laion/clap-htsat-unfused"
        )

    @torch.no_grad()
    def encode_segment(self, audio_path: str, start: float, duration: float = 1.0):
        audio, sr = librosa.load(audio_path, sr=48000,
                                  offset=start, duration=duration)
        inputs = self.processor(
            audios=audio, sampling_rate=sr, return_tensors="pt"
        ).to(self.device)
        emb = self.model.get_audio_features(**inputs)
        return emb / emb.norm(dim=-1, keepdim=True)  # L2 normalize


class CLIPPatchEncoder:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load ViT-L/14 — gives 16x16=256 patch tokens
        self.model, self.preprocess = clip.load("ViT-L/14", device=self.device)
        self.model.eval()
        self.patch_size = 16  # number of patches per side

    @torch.no_grad()
    def encode_patches(self, frame_bgr: np.ndarray):
        """Returns patch embeddings shape [patch_h, patch_w, 512]"""
        img = Image.fromarray(frame_bgr[:, :, ::-1])  # BGR→RGB
        x = self.preprocess(img).unsqueeze(0).to(self.device)

        patch_tokens = {}
        def hook(module, input, output):
            # output shape: [1, n_patches+1, d] — drop CLS token
            patch_tokens['feats'] = output[:, 1:, :]

        handle = self.model.visual.transformer.resblocks[-1].register_forward_hook(hook)
        _ = self.model.encode_image(x)
        handle.remove()

        feats = patch_tokens['feats'].squeeze(0)  # [256, 1024]
        # Project to 512 via model's projection layer
        feats = feats @ self.model.visual.proj  # [256, 512]
        feats = feats / feats.norm(dim=-1, keepdim=True)
        # Reshape to spatial grid
        return feats.view(self.patch_size, self.patch_size, -1)