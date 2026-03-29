import cv2
import numpy as np

def overlay_heatmap(frame_bgr, sim_map, alpha=0.5):
    height, width = frame_bgr.shape[:2]
    # Upsample similarity map to frame size
    heatmap = cv2.resize(sim_map.astype(np.float32), (width, height),
                          interpolation=cv2.INTER_LINEAR)
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    return cv2.addWeighted(frame_bgr, 1 - alpha, heatmap_color, alpha, 0)