import torch
import torch.nn as nn

class AudioAttention(nn.Module):
    def __init__(self, audio_dim, feat_channels):
        super().__init__()
        self.attn_fc = nn.Sequential(
            nn.Linear(audio_dim, feat_channels),
            nn.Tanh(),
            nn.Linear(feat_channels, feat_channels)
        )

    def forward(self, visual_feat, audio_emb):
        mask = self.attn_fc(audio_emb)
        mask = mask.unsqueeze(-1).unsqueeze(-1)
        return visual_feat * (1 + mask)
