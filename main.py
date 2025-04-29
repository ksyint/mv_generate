import torch
from model import AudioConditionalUNet
from trainer import train
from evaluation import compute_avs_score
from utils import save_video
from dataset import MusicVideoDataset
from audio_preprocess import load_audio_mel
import numpy as np

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AudioConditionalUNet(img_ch=3, audio_dim=80, base_ch=64, timesteps=1000)
    dataset = MusicVideoDataset('data/')
    train(model, dataset, epochs=10, lr=1e-4, device=device)
    wav, _ = dataset[0]
    model.eval()
    audio_mel = load_audio_mel(wav).to(device)
    x = torch.randn(1,3,256,256).to(device)
    frames = []
    with torch.no_grad():
        for t in reversed(range(model.scheduler.timesteps)):
            x = model(x, audio_mel, torch.tensor([t], device=device))
        frames = x.cpu().permute(0,2,3,1).numpy()
    save_video(list(frames), 'result.mp4', fps=24)
    avs = compute_avs_score(wav.numpy(), np.stack(frames))
    print(f"AV Synchrony Score: {avs:.4f}")

if __name__ == '__main__':
    main()
