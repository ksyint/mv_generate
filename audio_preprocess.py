import librosa
import numpy as np
import torch

def load_audio_mel(path, sr=22050, n_fft=1024, hop_length=512, n_mels=80):
    y, sr = librosa.load(path, sr=sr)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft,
        hop_length=hop_length, n_mels=n_mels,
        power=2.0, norm='slaney', htk=True
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return torch.from_numpy(mel_db).float()
