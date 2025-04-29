import torch
import torch.nn as nn
from scheduler import DiffusionScheduler
from attention import AudioAttention

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(32, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1)
        )
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch!=out_ch else nn.Identity()

    def forward(self, x):
        return self.shortcut(x) + self.conv(x)

class AudioConditionalUNet(nn.Module):
    def __init__(self, img_ch=3, audio_dim=80, base_ch=64, timesteps=1000):
        super().__init__()
        self.scheduler = DiffusionScheduler(timesteps)
        self.enc1 = nn.Sequential(
            nn.Conv2d(img_ch, base_ch, 3, padding=1),
            ResidualBlock(base_ch, base_ch)
        )
        self.enc2 = nn.Sequential(
            ResidualBlock(base_ch, base_ch*2),
            ResidualBlock(base_ch*2, base_ch*2)
        )
        self.mid = ResidualBlock(base_ch*2, base_ch*2)
        self.attn = AudioAttention(audio_dim, base_ch*2)
        self.dec2 = nn.Sequential(
            ResidualBlock(base_ch*4, base_ch*2),
            ResidualBlock(base_ch*2, base_ch*2)
        )
        self.dec1 = nn.Sequential(
            ResidualBlock(base_ch*2 + base_ch, base_ch),
            ResidualBlock(base_ch, base_ch)
        )
        self.out_conv = nn.Conv2d(base_ch, img_ch, 1)
        self.audio_proj = nn.Linear(audio_dim, base_ch*2)

    def forward(self, x, audio_emb, t):
        e1 = self.enc1(x)
        e2 = self.enc2(nn.functional.avg_pool2d(e1, 2))
        m = self.mid(nn.functional.avg_pool2d(e2, 2))
        m = self.attn(m, audio_emb)
        d2 = self.dec2(torch.cat([m, e2], dim=1))
        d2 = nn.functional.interpolate(d2, scale_factor=2)
        d1 = self.dec1(torch.cat([d2, e1], dim=1))
        out = self.out_conv(nn.functional.interpolate(d1, scale_factor=2))
        return out
