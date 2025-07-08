import torch
import math
import gpytorch

class StructuredORFFeatureMap(torch.nn.Module):
    def __init__(self, input_dim, num_features):
        super().__init__()
        self.num_features = num_features
        self.input_dim = input_dim
        self.block_size = input_dim * 2   #2*d rule of thumb, from a paper I saw
        self.num_blocks = num_features // self.block_size

        self.log_lengthscale = torch.nn.Parameter(torch.tensor(0.0)) 

        
        W_blocks = []                                   #break matrix intro blocks
        for _ in range(self.num_blocks):
            G = torch.randn(self.block_size,input_dim)  #small sample
            Q, R = torch.linalg.qr(G)                   #small qr for efficiency, slight redundancy but better computational speed
            W_blocks.append(Q)                          #append to list
        
        W = torch.cat(W_blocks,dim=0)[:num_features]    #add up all orthogonal vectors
        self.W_raw = torch.nn.Parameter(W.T) 
        self.register_buffer("b", 2 * math.pi * torch.rand(W.shape[0])) #truncates B incase num_features % (input_dim*2) !=0

    def forward(self, x):
        lengthscale = torch.exp(self.log_lengthscale)
        W = self.W_raw / lengthscale
        projection = x @ W + self.b
        return math.sqrt(2.0 / self.num_features) * torch.cos(projection)


# Custom GP model using Linear Kernel on ORF
class StructuredORFGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_orf_features=100,input_dim=2):
        super().__init__(train_x, train_y, likelihood)
        self.feature_map = StructuredORFFeatureMap(input_dim=input_dim, num_features=num_orf_features)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.LinearKernel()

    def forward(self, x):
        features = self.feature_map(x)
        mean_x = self.mean_module(features)
        covar_x = self.covar_module(features)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    