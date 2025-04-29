import cv2
import numpy as np

def save_video(frames, path, fps=30):
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for f in frames:
        if f.dtype!=np.uint8:
            f = ((f - f.min())/(f.max()-f.min())*255).astype(np.uint8)
        writer.write(f)
    writer.release()
