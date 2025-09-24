import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Defined.diffusionModels.sinusoidalEmbedding import SinusoidalEmbeddingModule
import math

class ResidualMLP(nn.Module): # Residual block for MLP
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.SiLU()
    
    def forward(self, x):
        return x + self.fc2(self.act(self.norm(self.fc1(x)))) / math.sqrt(2)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128,time_dim=32, num_res_blocks=2):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalEmbeddingModule(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        self.input_layer = nn.Linear(input_dim + time_dim, hidden_dim)
        self.res_blocks = nn.ModuleList([ResidualMLP(hidden_dim) for _ in range(num_res_blocks)])
        self.out = nn.Linear(hidden_dim, input_dim) #input dim must = output dim for sampling and denoising process

    def forward(self, x, t):
        t_emb = self.time_mlp(t.float())
        h = torch.cat([x, t_emb], dim=-1)
        h = self.input_layer(h)
        for block in self.res_blocks:
            h = block(h)
        return self.out(h)