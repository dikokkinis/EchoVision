import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SoundLocalizer:
    def __init__(self, audio_enc, clip_enc):
        self.audio_enc = audio_enc
        self.clip_enc = clip_enc
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        audio_dim = 512
        patch_dim = 768
        self.proj = nn.Linear(audio_dim, patch_dim, bias=False).to(audio_enc.device)
        nn.init.eye_( 
            self.proj.weight[:audio_dim, :audio_dim]
            if patch_dim >= audio_dim
            else self.proj.weight
        ) 

    @torch.no_grad()
    def localize(self, audio_path: str, frame_bgr, timestamp: float):
        audio_emb = self.audio_enc.encode_segment(audio_path, start=timestamp)
        # [1, 512] → [1, 768]
        audio_emb = self.proj(torch.tensor(audio_emb).to(self.audio_enc.device))
        audio_emb = F.normalize(audio_emb, p=2, dim=-1)

        patch_embs = self.clip_enc.encode_patches(frame_bgr)  # [16, 16, 768]
        patch_embs = patch_embs.to(self.clip_enc.device)
        a = audio_emb.squeeze(0)               # [768]
        p = patch_embs.view(-1, 768)           # [256, 768]
        sim = (p @ a).view(16, 16)             # [16, 16]

        sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-8)
        return sim.detach().cpu().numpy()