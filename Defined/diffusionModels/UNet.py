import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Defined.diffusionModels.sinusoidalEmbedding import SinusoidalEmbeddingModule


class ResidualBlock(nn.Module): #Stabilzes training, allows for deeper networks
    def __init__(self, in_channels, out_channels, time_dim=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
        if time_dim is not None:
            self.time_mlp = nn.Linear(time_dim, out_channels)
        else:
            self.time_mlp = None
        
    def forward(self, x, t_emb=None):
        out = self.relu(self.norm1(self.conv1(x)))
        
        if self.time_mlp is not None and t_emb is not None:
            # Shape: (B, C) -> (B, C, 1, 1) so it can broadcast
            time_out = self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
            out = out + time_out
        
        out = self.norm2(self.conv2(out))
        out += self.skip(x)
        return self.relu(out)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_dim=64, dim_mults=(1,2,4,8), time_dim=256):
        super().__init__()
        dims = [base_dim * m for m in dim_mults]

        #Timestep embedding MLP
        self.time_mlp = nn.Sequential(
            SinusoidalEmbeddingModule(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.ReLU(),
            nn.Linear(time_dim * 4, time_dim)
        )

        # Encoder
        self.encoders = nn.ModuleList()
        prev_dim = in_channels
        for d in dims:
            self.encoders.append(ResidualBlock(prev_dim, d, time_dim=time_dim))
            prev_dim = d
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ResidualBlock(dims[-1], dims[-1]*2, time_dim=time_dim)

        # Decoder
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        reversed_dims = list(reversed(dims))
        prev_dim = dims[-1]*2
        for d in reversed_dims:
            self.upconvs.append(nn.ConvTranspose2d(prev_dim, d, 2, stride=2))
            self.decoders.append(ResidualBlock(prev_dim, d,time_dim=time_dim))
            prev_dim = d

        self.out = nn.Conv2d(prev_dim, out_channels, 1)
    
    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        
        enc_feats = []

        # Encoder
        for enc in self.encoders:
            x = enc(x, t_emb)   #pass time-step embedding
            enc_feats.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x,t_emb)

        # Decoder
        for up, dec in zip(self.upconvs, self.decoders):
            x = up(x)
            skip = enc_feats.pop()
            # Crop if necessary
            if x.shape[-2:] != skip.shape[-2:]:
                diffY = skip.size(2) - x.size(2)
                diffX = skip.size(3) - x.size(3)
                x = F.pad(x, [diffX//2, diffX - diffX//2,
                              diffY//2, diffY - diffY//2])
            x = dec(torch.cat([x, skip], dim=1), t_emb)

        return self.out(x)