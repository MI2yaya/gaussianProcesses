import torch
import math
import gpytorch


class RFFFeatureMap(torch.nn.Module):
    def __init__(self, input_dim, num_features):
        super().__init__()
        self.num_features = num_features
        self.input_dim = input_dim

        # Learnable log lengthscale for stability
        self.log_lengthscale = torch.nn.Parameter(torch.tensor(0.0))  # exp(0) = 1

        # Fixed random frequencies and phases
        self.register_buffer("W_raw", torch.randn(input_dim, num_features))
        self.register_buffer("b", 2 * math.pi * torch.rand(num_features))

    def forward(self, x):
        lengthscale = torch.exp(self.log_lengthscale)
        W = self.W_raw / lengthscale
        projection = x @ W + self.b
        return math.sqrt(2.0 / self.num_features) * torch.cos(projection)


# Custom GP model using Linear Kernel on RFF
class RFFGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_rff_features=100,input_dim=1):
        super().__init__(train_x, train_y, likelihood)
        self.feature_map = RFFFeatureMap(input_dim=input_dim, num_features=num_rff_features)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.LinearKernel()

    def forward(self, x):
        features = self.feature_map(x)
        mean_x = self.mean_module(features)
        covar_x = self.covar_module(features)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    