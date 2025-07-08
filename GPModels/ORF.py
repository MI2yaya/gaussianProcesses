import torch
import math
import gpytorch

class ORFFeatureMap(torch.nn.Module):
    def __init__(self, input_dim, num_features):
        super().__init__()
        self.num_features = num_features
        self.input_dim = input_dim

        # Learnable log lengthscale for stability
        self.log_lengthscale = torch.nn.Parameter(torch.tensor(0.01))  # exp(0) = 1

        # Fixed random frequencies and phases
        G = torch.randn(num_features, input_dim)  # generate random gaussian matrix
        Q, R = torch.linalg.qr(G)                 # QR decomposition to get orthogonal Q (R is upper triangular)
        self.W_raw = torch.nn.Parameter(Q.T * math.sqrt(self.input_dim))    #trainable W, woah, used to better spectral elements
        self.register_buffer("b", 2 * math.pi * torch.rand(num_features))

    def forward(self, x):
        lengthscale = torch.exp(self.log_lengthscale)
        W = self.W_raw / lengthscale
        projection = x @ W + self.b
        return math.sqrt(2.0 / self.num_features) * torch.cos(projection)


# Custom GP model using Linear Kernel on ORF
class ORFGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_orf_features=100,input_dim=2):
        super().__init__(train_x, train_y, likelihood)
        self.feature_map = ORFFeatureMap(input_dim=input_dim, num_features=num_orf_features)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.LinearKernel()

    def forward(self, x):
        features = self.feature_map(x)
        mean_x = self.mean_module(features)
        covar_x = self.covar_module(features)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    