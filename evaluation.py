import numpy as np

def compute_avs_score(audio_signal, video_frames):
    audio_env = np.mean(np.abs(audio_signal), axis=1)
    vid_diff = np.mean(np.abs(np.diff(video_frames, axis=0)), axis=(1,2,3))
    L = min(len(audio_env), len(vid_diff))
    if np.std(audio_env[:L])<1e-6 or np.std(vid_diff[:L])<1e-6:
        return 0.0
    score = np.corrcoef(audio_env[:L], vid_diff[:L])[0,1]
    return float(score)
