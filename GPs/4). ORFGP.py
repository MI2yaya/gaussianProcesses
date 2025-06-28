import math
import torch
import gpytorch
from matplotlib import pyplot as plt


train_x = torch.linspace(0, 1, 100)
epsilon = torch.randn(train_x.size()) * math.sqrt(0.04) #noise
train_y = torch.sin(train_x * (3 * math.pi)) + epsilon

plt.figure(figsize=(8, 4))
plt.scatter(train_x.numpy(), train_y.numpy(), label='Training Data', s=10)
plt.title("Toy Training Data: y = sin(3x) + noise")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()


class ORFFeatureMap(torch.nn.Module):
    def __init__(self, input_dim, num_features):
        super().__init__()
        self.num_features = num_features
        self.input_dim = input_dim

        # Learnable log lengthscale for stability
        self.log_lengthscale = torch.nn.Parameter(torch.tensor(0.0))  # exp(0) = 1

        # Fixed random frequencies and phases
        G = torch.randn(num_features, input_dim)  # note dimensions flipped for QR
        Q, R = torch.linalg.qr(G)                 # QR decomposition to get orthogonal Q
        self.register_buffer("W_raw", Q.T)        # transpose to shape (input_dim, num_features)
        self.register_buffer("b", 2 * math.pi * torch.rand(num_features))

    def forward(self, x):
        lengthscale = torch.exp(self.log_lengthscale)
        W = self.W_raw / lengthscale
        projection = x @ W + self.b
        return math.sqrt(2.0 / self.num_features) * torch.cos(projection)


# Custom GP model using Linear Kernel on RFF
class RFFGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_rff_features=100):
        super().__init__(train_x, train_y, likelihood)
        self.feature_map = ORFFeatureMap(input_dim=1, num_features=num_rff_features)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.LinearKernel()

    def forward(self, x):
        features = self.feature_map(x)
        mean_x = self.mean_module(features)
        covar_x = self.covar_module(features)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = RFFGPModel(train_x, train_y, likelihood, num_rff_features=500)

# Train
model.train()
likelihood.train()

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(50):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    print(f"Iter {i+1}/50 - Loss: {loss.item():.3f}   noise: {model.likelihood.noise.item():.3f}")
    optimizer.step()

# Predict
model.eval()
likelihood.eval()

test_x = torch.linspace(0, 1, 51).unsqueeze(-1)
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    preds = likelihood(model(test_x))

# Plot
f, ax = plt.subplots(1, 1, figsize=(6, 4))
lower, upper = preds.confidence_region()
ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
ax.plot(test_x.numpy(), preds.mean.numpy(), 'b')
ax.fill_between(test_x.squeeze().numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
ax.set_ylim([-3, 3])
ax.legend(['Observed Data', 'Mean', 'Confidence'])
plt.show()