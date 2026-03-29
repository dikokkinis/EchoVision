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

    def localize(self, audio_path, frame_bgr, timestamp):
        audio_emb = self.audio_enc.encode_segment(
            audio_path, start=timestamp
        ) 
        print("Audio embedding shape:", audio_emb.shape)  # Debug print
        audio_emb = self.proj(audio_emb) 
        audio_emb = F.normalize(audio_emb, p=2, dim=-1)

        patch_embs = self.clip_enc.encode_patches(frame_bgr) 
        print("Patch embeddings shape:", patch_embs.shape)  # Debug print

        # Cosine similarity
        a = audio_emb.squeeze(0) 
        print("Audio embedding shape:", a.shape)

        p = patch_embs.view(-1, patch_embs.shape[-1])      
        print("Patch embeddings reshaped for similarity:", p.shape)   

        sim = (p @ a).view(16, 16)                       

        # Normalize to [0, 1] for heatmap visualization
        sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-8)
        return sim.cpu().detach().numpy()