import torch
from torch.utils.data import TensorDataset, DataLoader
from data.MNIST.mnistDatasetLoader import MnistDataloader
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Defined.diffusionModels.Trainer import DiffusionTrainer
from Defined.diffusionModels.GaussianDiffusion import GaussianDiffusion
from Defined.diffusionModels.UNet import UNet

# Load MNIST
mnist_dataloader = MnistDataloader()
(x_train, y_train), _ = mnist_dataloader.load_data()
x_train = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1) / 255.0 * 2 - 1
y_train = torch.tensor(y_train, dtype=torch.long)

batch_size=32

dataset = TensorDataset(x_train, y_train)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("using cuda:", torch.cuda.is_available())

model = UNet(
    in_channels=1,
    out_channels=1,
    base_dim=64,
    dim_mults=(1, 2, 4)
).to(device)
diffusion = GaussianDiffusion(
    model, 
    timesteps=1000,
    schedule="cosine"
).to(device)
trainer = DiffusionTrainer(
    model, 
    diffusion, 
    dataset, 
    batch_size=batch_size, 
    lr=1e-4,
    device=device,
    ema_decay=0.995,
    patience=10,
    ckpt_path=os.path.join('diffusionModels\data\MNIST',"best_ema.pt")
)

print("Starting training...")
trainer.train(steps=10000, log_every=100)

print("Sampling images...")
sampled_images = trainer.sample(num_samples=16,use_ema=True)

import matplotlib.pyplot as plt

# Plot
fig, axes = plt.subplots(4, 4, figsize=(6,6))
for i, ax in enumerate(axes.flat):
    img = sampled_images[i].detach().cpu().numpy().transpose(1, 2, 0)
    img = (img + 1) / 2  # rescale 0-1
    ax.imshow(img.squeeze(), cmap="gray")
    ax.axis("off")

plt.show()