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
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.act = nn.SiLU()
    
    def forward(self, x):
        h = self.fc1(self.norm1(x))
        h = self.act(h)
        h = self.fc2(self.norm2(h))
        return x + h / math.sqrt(2)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128,time_dim=32, num_res_blocks=2):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalEmbeddingModule(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.res_blocks = nn.ModuleList([ResidualMLP(hidden_dim) for _ in range(num_res_blocks)])
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, input_dim) #more layers can be added here
        )

    def forward(self, x, t):
        t_emb = self.time_mlp(t.float())   # [B, hidden_dim]
        h = self.input_layer(x)            # [B, hidden_dim]
        for block in self.res_blocks:
            h = block(h + t_emb)           # inject timestep info
        return self.out(h)