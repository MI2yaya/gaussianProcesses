import torch
import math
import torch.nn as nn
def sinusoidal_embedding(timesteps, dim) -> torch.Tensor: #sinusoidal time-step embedding, does not need input range
    half_dim = dim // 2
    device = timesteps.device
    # frequency scales
    freqs = torch.exp(
        -math.log(10000) * torch.arange(0, half_dim, dtype=torch.float32, device=device) / half_dim
    )
    # (batch, half_dim)
    args = timesteps[:, None].float() * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return emb

class SinusoidalEmbeddingModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t shape: (batch,)
        return sinusoidal_embedding(t, self.dim)