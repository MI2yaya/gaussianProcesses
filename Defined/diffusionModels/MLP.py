import torch
import torch.nn as nn
from sinusoidalEmbedding import sinusoidal_embedding

class ResidualMLP(nn.Module): # Residual block for MLP
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.ReLU()
    
    def forward(self, x):
        return x + self.fc2(self.act(self.fc1(x)))

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, time_dim=32, num_res_blocks=2):
        super().__init__()
        self.time_mlp = nn.Sequential(
            sinusoidal_embedding,  # or use Linear if you want
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        self.input_layer = nn.Linear(input_dim + time_dim, hidden_dim)
        self.res_blocks = nn.ModuleList([ResidualMLP(hidden_dim) for _ in range(num_res_blocks)])
        self.out = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, t):
        t_emb = self.time_mlp(t.unsqueeze(-1).float())
        h = torch.cat([x, t_emb], dim=-1)
        h = self.input_layer(h)
        for block in self.res_blocks:
            h = block(h)
        return self.out(h)