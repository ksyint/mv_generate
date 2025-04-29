import torch
from torch.utils.data import DataLoader
from audio_preprocess import load_audio_mel

def train(model, dataset, epochs, lr, device):
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    for ep in range(epochs):
        for wavs, frames in loader:
            wavs, frames = wavs.to(device), frames.to(device)
            audio_mel = load_audio_mel(wavs)
            t = torch.randint(0, model.scheduler.timesteps, (wavs.size(0),), device=device)
            noise = torch.randn_like(frames)
            alpha_bar_t = torch.from_numpy(model.scheduler.alpha_bar[t]).to(device)
            noisy = alpha_bar_t[:,None,None,None].sqrt() * frames + \
                    (1-alpha_bar_t)[:,None,None,None].sqrt() * noise
            pred = model(noisy, audio_mel, t)
            loss = torch.mean((pred - noise)**2)
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"[Epoch {ep+1}/{epochs}] Loss: {loss.item():.4f}")
