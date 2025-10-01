import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Defined.diffusionModels.sinusoidalEmbedding import SinusoidalEmbeddingModule

def get_conv(dim, in_ch, out_ch, k, stride=1, padding=0):
    if dim == 1: return nn.Conv1d(in_ch, out_ch, k, stride, padding)
    if dim == 2: return nn.Conv2d(in_ch, out_ch, k, stride, padding)
    if dim == 3: return nn.Conv3d(in_ch, out_ch, k, stride, padding)
    raise ValueError(f"Unsupported dim {dim}")

def get_norm(dim, ch):
    if dim == 1: return nn.BatchNorm1d(ch)
    if dim == 2: return nn.BatchNorm2d(ch)
    if dim == 3: return nn.BatchNorm3d(ch)
    raise ValueError(f"Unsupported dim {dim}")

def get_pool(dim):
    if dim == 1: return nn.MaxPool1d(2)
    if dim == 2: return nn.MaxPool2d(2)
    if dim == 3: return nn.MaxPool3d(2)
    raise ValueError(f"Unsupported dim {dim}")

def get_upsample(dim, in_ch, out_ch):
    if dim == 1: return nn.ConvTranspose1d(in_ch, out_ch, 2, stride=2)
    if dim == 2: return nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
    if dim == 3: return nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2)
    raise ValueError(f"Unsupported dim {dim}")


class ResidualBlock(nn.Module): #Stabilzes training, allows for deeper networks
    def __init__(self, dim, in_ch, out_ch, time_dim=None):
        super().__init__()
        self.conv1 = get_conv(dim, in_ch, out_ch, 3, padding=1)
        self.norm1 = get_norm(dim, out_ch)
        self.conv2 = get_conv(dim, out_ch, out_ch, 3, padding=1)
        self.norm2 = get_norm(dim, out_ch)

        self.skip = get_conv(dim, in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

        self.time_mlp = nn.Linear(time_dim, out_ch) if time_dim is not None else None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, t_emb=None):
        h = self.relu(self.norm1(self.conv1(x)))

        if self.time_mlp is not None and t_emb is not None:
            time_out = self.time_mlp(t_emb)
            while len(time_out.shape) < len(h.shape):
                time_out = time_out.unsqueeze(-1)
            h = h + time_out

        h = self.norm2(self.conv2(h))
        return self.relu(h + self.skip(x))

class UNet(nn.Module):
    def __init__(self, dim, in_ch=1, out_ch=1,base_dim=32,dim_mults=(1,2,4,8), time_dim=256):
        super().__init__()
        self.dim = dim
        self.in_ch = in_ch

        #Timestep embedding MLP
        self.time_mlp = nn.Sequential(
            SinusoidalEmbeddingModule(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.ReLU(),
            nn.Linear(time_dim * 4, time_dim)
        )

        # Encoder
        dims = [base_dim * m for m in dim_mults]
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev_ch = in_ch
        for d in dims:
            self.encoders.append(ResidualBlock(dim, prev_ch, d, time_dim=time_dim))
            self.pools.append(get_pool(dim))
            prev_ch = d


        # Bottleneck
        self.bottleneck = ResidualBlock(dim, dims[-1], dims[-1]*2, time_dim=time_dim)

        # Decoder
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        reversed_dims = list(reversed(dims))
        prev_ch = dims[-1]*2
        for d in reversed_dims:
            self.upconvs.append(get_upsample(dim, prev_ch, d))
            self.decoders.append(ResidualBlock(dim, 2*d, d, time_dim=time_dim))
            prev_ch = d

        self.out = get_conv(dim,prev_ch,out_ch,1)
    
    def forward(self, x, t):
        assert x.dim() == 3, f"Expected (B, C, T), got {x.shape}"
        assert x.size(1) == self.in_ch, f"Expected in_ch={self.in_ch}, got {x.size(1)}"
        t_emb = self.time_mlp(t)
        
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x, t_emb)
            skips.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x,t_emb)

        # Decoder
        skips = skips[::-1]
        for up, dec, skip in zip(self.upconvs, self.decoders, skips):
            x = up(x)
            if x.shape[-1] != skip.shape[-1]:
                x = F.pad(x, (0, skip.shape[-1] - x.shape[-1]))
            x = torch.cat((skip, x), dim=1) #dim=skip_ch+up_ch
            x = dec(x, t_emb)

        return self.out(x)