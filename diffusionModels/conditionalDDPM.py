import torch
from torch.utils.data import TensorDataset, DataLoader
from data.MNIST.mnistDatasetLoader import MnistDataloader
import sys, os
import matplotlib.pyplot as plt
from PIL import Image
import math
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Defined.diffusionModels.Trainer import DiffusionTrainer
from Defined.diffusionModels.GaussianDiffusion import GaussianDiffusion
from Defined.diffusionModels.UNet import UNet

import torch.nn.functional as F

mnist_dataloader = MnistDataloader()
(x_train, y_train), _ = mnist_dataloader.load_data()
x_train = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1) / 255.0 * 2 - 1
y_train = torch.tensor(y_train, dtype=torch.long)

batch_size=32

dataset = TensorDataset(x_train, y_train)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("using cuda:", torch.cuda.is_available())

ckpt_path = os.path.join('diffusionModels','data','MNIST','best_ema_inference.pt')

preload = True
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
    use_EMA=True,
    ckpt_path=ckpt_path
)

if preload:
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    
else:
    trainer.train(
        steps=5000,
        log_every=100
    )


def A(x, kernel):
    return F.conv2d(x, kernel, padding="same")

def A_T(y, kernel):
    # flip kernel horizontally + vertically
    flipped = torch.flip(kernel, [2,3])
    return F.conv2d(y, flipped, padding="same")

def get_gaussian_kernel(kernel_size=5, sigma=1.0, device="cpu"):
    ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = torch.meshgrid([ax, ax], indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, kernel_size, kernel_size).to(device)
    return kernel

x_gt = dataset[0][0].unsqueeze(0).to(device)  # clean ground truth image


kernel = get_gaussian_kernel(kernel_size=5, sigma=1.0, device=device)
kernel = kernel / kernel.sum()
y_meas = A(x_gt, kernel)

restored = diffusion.posterior_sample(
    y_meas=y_meas,
    A=lambda z: A(z, kernel),
    A_T=lambda z: A_T(z, kernel),
    lam=.1,
    sigma_y=1  # adjust to approximate measurement noise
)


import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(9, 3))
axes[0].imshow(x_gt[0,0].cpu(), cmap="gray"); axes[0].set_title("Ground Truth"); axes[0].axis("off")
axes[1].imshow(y_meas[0,0].cpu(), cmap="gray"); axes[1].set_title("Blurred"); axes[1].axis("off")
axes[2].imshow(restored[0,0].cpu(), cmap="gray"); axes[2].set_title("Restored"); axes[2].axis("off")
plt.show()