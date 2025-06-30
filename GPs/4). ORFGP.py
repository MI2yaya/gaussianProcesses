import math
import torch
import gpytorch
from matplotlib import pyplot as plt


grid_size = 30
x1 = torch.linspace(0, 1, grid_size)
x2 = torch.linspace(0, 1, grid_size)
X1, X2 = torch.meshgrid(x1, x2, indexing='ij')
train_x = torch.stack([X1.reshape(-1), X2.reshape(-1)], dim=-1) 

noise_std = 0.1
train_y = torch.sin(3 * math.pi * train_x[:, 0]) * torch.cos(2 * math.pi * train_x[:, 1]) + noise_std * torch.randn(train_x.size(0))


plt.scatter(train_x[:, 0].numpy(), train_x[:, 1].numpy(), c=train_y.numpy(), cmap='viridis')
plt.colorbar(label='y')
plt.title("Training Data: sin(3πx1) * cos(2πx2) + noise")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

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
        self.W_raw = torch.nn.Parameter(Q.T * math.sqrt(self.input_dim))    #trainable W, woah
        self.register_buffer("b", 2 * math.pi * torch.rand(num_features))

    def forward(self, x):
        lengthscale = torch.exp(self.log_lengthscale)
        W = self.W_raw / lengthscale
        projection = x @ W + self.b
        return math.sqrt(2.0 / self.num_features) * torch.cos(projection)


# Custom GP model using Linear Kernel on ORF
class RFFGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_orf_features=100):
        super().__init__(train_x, train_y, likelihood)
        self.feature_map = ORFFeatureMap(input_dim=2, num_features=num_orf_features)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.LinearKernel()

    def forward(self, x):
        features = self.feature_map(x)
        mean_x = self.mean_module(features)
        covar_x = self.covar_module(features)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    
    
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = RFFGPModel(train_x, train_y, likelihood, num_orf_features=100) #only 100 features is preposterous

# Train
model.train()
likelihood.train()

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

trials=150
for i in range(trials):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    print(f"Iter {i+1}/{trials} - Loss: {loss.item():.3f}   noise: {model.likelihood.noise.item():.3f}")
    optimizer.step()
    if loss.item() < 0.05:
        print("Stopping early: loss below 0.05")
        break

# Predict
model.eval()
likelihood.eval()

test_grid_size = 30
test_x1 = torch.linspace(0, 1, test_grid_size)
test_x2 = torch.linspace(0, 1, test_grid_size)
T1, T2 = torch.meshgrid(test_x1, test_x2, indexing='ij')
test_x = torch.stack([T1.reshape(-1), T2.reshape(-1)], dim=-1)

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    preds = likelihood(model(test_x))

true_y = torch.sin(3 * math.pi * test_x[:, 0]) * torch.cos(2 * math.pi * test_x[:, 1])
true_y_grid = true_y.reshape(test_grid_size, test_grid_size).numpy()

mean = preds.mean  # shape: (test_grid_size^2,)
lower, upper = preds.confidence_region()  # same shape, lower and upper 95% CI bounds

# Reshape mean and bounds back to grid shape for plotting
mean_grid = mean.reshape(test_grid_size, test_grid_size).numpy()
lower_grid = lower.reshape(test_grid_size, test_grid_size).numpy()
upper_grid = upper.reshape(test_grid_size, test_grid_size).numpy()

#3d plotting :)
fig = plt.figure(figsize=(14, 6))

#true function
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(T1.numpy(), T2.numpy(), true_y_grid, cmap='viridis', edgecolor='none')
ax1.set_title("True Function")
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
ax1.set_zlabel('y')

#predicted
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(T1.numpy(), T2.numpy(), mean_grid, cmap='viridis', edgecolor='none')
ax2.set_title("GP Predicted Mean")
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
ax2.set_zlabel('y')

plt.tight_layout()
plt.show()