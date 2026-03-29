import torch
from torch.linalg import norm
import torch.nn.functional as F
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
            audio=audio, sampling_rate=sr, return_tensors="pt"
        ).to(self.device)

        outputs = self.model.get_audio_features(**inputs)
        audio_emb = outputs.pooler_output  # [1, 512]
        audio_emb = F.normalize(audio_emb, p=2, dim=-1) # L2 normalize
        return audio_emb

class CLIPPatchEncoder:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load ViT-L/14 — gives 16x16=256 patch tokens
        self.model, self.preprocess = clip.load("ViT-L/14", device=self.device)
        self.model.eval()
        self.patch_size = 16  # number of patches per side
        self.embed_dim = 768  

    @torch.no_grad()
    def encode_patches(self, frame_bgr: np.ndarray):
        """Returns patch embeddings shape"""
        img = Image.fromarray(frame_bgr[:, :, ::-1])  # BGR→RGB
        x = self.preprocess(img).unsqueeze(0).to(self.device)

        patch_tokens = {}
        def hook(module, input, output):
            # output shape: [1, n_patches+1, d] — drop CLS token
            tokens = output[0] if isinstance(output, tuple) else output
            patch_tokens['x'] = tokens[1:, :, :]

        handle = self.model.visual.transformer.resblocks[-1].register_forward_hook(hook)
        self.model.encode_image(x)
        handle.remove()

        feats = patch_tokens['x'].squeeze(1)

        # Project 1024 -> 768 
        if self.model.visual.proj is not None:
            feats = feats @ self.model.visual.proj  # [256, 768]

        feats = F.normalize(feats, p=2, dim=-1)  # L2 normalize
        # Reshape to spatial grid
        return feats.view(self.patch_size, self.patch_size, -1)