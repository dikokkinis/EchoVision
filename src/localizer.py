import torch
import torch.nn.functional as F
import numpy as np
import cv2
import numpy as np

class SoundLocalizer:
    def __init__(self, audio_enc, clip_enc):
        self.audio_enc = audio_enc
        self.clip_enc = clip_enc

    def localize(self, audio_path, frame_bgr, timestamp):
        audio_emb = self.audio_enc.encode_segment(
            audio_path, start=timestamp
        )  # [1, 512]
        patch_embs = self.clip_enc.encode_patches(frame_bgr)  # [16, 16, 512]

        # Cosine similarity
        a = audio_emb.squeeze(0)                          # [512]
        p = patch_embs.view(-1, 512)                      # [256, 512]
        sim = (p @ a).view(16, 16)                        # [16, 16]

        # Normalize to [0, 1]
        sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-8)
        return sim.cpu().numpy()

def overlay_heatmap(frame_bgr, sim_map, alpha=0.5):
    height, width = frame_bgr.shape[:2]
    # Upsample similarity map to frame size
    heatmap = cv2.resize(sim_map.astype(np.float32), (width, height),
                          interpolation=cv2.INTER_LINEAR)
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    return cv2.addWeighted(frame_bgr, 1 - alpha, heatmap_color, alpha, 0)