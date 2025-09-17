import torch
from torch.utils.data import TensorDataset, DataLoader
from data.MNIST.mnistDatasetLoader import MnistDataloader
import sys, os
import matplotlib.pyplot as plt
from PIL import Image
import math
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Defined.diffusionModels.Trainer import DiffusionTrainer
from Defined.diffusionModels.GaussianDiffusion import GaussianDiffusion
from Defined.diffusionModels.UNet import UNet

import torch.nn.functional as F
import random

random.seed(42)

save=True
save_dir = os.path.join("diffusionModels", "figs")
os.makedirs(save_dir, exist_ok=True)

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


model.eval()

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

prior_blurred_image = dataset[random.randint(0, len(dataset) - 1)][0].unsqueeze(0).to(device)  # clean ground truth image


kernel = get_gaussian_kernel(kernel_size=5, sigma=1.0, device=device)
kernel = kernel / kernel.sum()
blurred_image = A(prior_blurred_image, kernel)

restored_blurred = diffusion.posterior_sample(
    y_meas=blurred_image,
    A=lambda z: A(z, kernel),
    A_T=lambda z: A_T(z, kernel),
    lam=.1,
    sigma_y=1
)

#corrupt

def corrupt_with_noise(x, noise_std=1.0, region=None):
    x_noisy = x.clone()
    if region is None:
        x_noisy += noise_std * torch.randn_like(x_noisy)
    else:
        i1, i2, j1, j2 = region
        x_noisy[:, :, i1:i2, j1:j2] += noise_std * torch.randn_like(x[:, :, i1:i2, j1:j2])
    return x_noisy


prior_corrupted_image = dataset[random.randint(0, len(dataset) - 1)][0].unsqueeze(0).to(device)
corrupted_image = corrupt_with_noise(prior_corrupted_image, noise_std=1.0, region=(0, 14, 0, 14))


mask = torch.ones_like(prior_corrupted_image)
mask[:,:,0:14,0:14] = 0.0   #portion unknown

def A(x): return mask * x 
def A_T(y): return mask * y

restored_inpaint = diffusion.posterior_sample(
    y_meas=corrupted_image,
    A=A,
    A_T=A_T,
    lam=0.1,
    sigma_y=1
)

#corrupt a lot

prior_corrupted_image2 = dataset[random.randint(0, len(dataset) - 1)][0].unsqueeze(0).to(device)
corrupted_image2 = corrupt_with_noise(prior_corrupted_image2, noise_std=.5)

mask = torch.ones_like(prior_corrupted_image2) * .5 #all somewhat unknown

restored_inpaint2 = diffusion.posterior_sample(
    y_meas=corrupted_image2,
    A=A,
    A_T=A_T,
    lam=0.1,
    sigma_y=1
)


fig, axes = plt.subplots(3, 3, figsize=(6, 6))
axes = axes.flatten() 
axes[0].imshow(prior_blurred_image[0,0].cpu(), cmap="gray"); axes[0].set_title("Ground Truth"); axes[0].axis("off")
axes[1].imshow(blurred_image[0,0].cpu(), cmap="gray"); axes[1].set_title("Blurred"); axes[1].axis("off")
axes[2].imshow(restored_blurred[0,0].cpu(), cmap="gray"); axes[2].set_title("Restored"); axes[2].axis("off")
axes[3].imshow(prior_corrupted_image[0,0].cpu(), cmap="gray"); axes[3].set_title("Ground Truth"); axes[3].axis("off")
axes[4].imshow(corrupted_image[0,0].cpu(), cmap="gray"); axes[4].set_title("Corrupted"); axes[4].axis("off")
axes[5].imshow(restored_inpaint[0,0].cpu(), cmap="gray"); axes[5].set_title("Restored"); axes[5].axis("off")
axes[6].imshow(prior_corrupted_image2[0,0].cpu(), cmap="gray"); axes[6].set_title("Ground Truth"); axes[6].axis("off")
axes[7].imshow(corrupted_image2[0,0].cpu(), cmap="gray"); axes[7].set_title("Very Corrupted"); axes[7].axis("off")
axes[8].imshow(restored_inpaint2[0,0].cpu(), cmap="gray"); axes[8].set_title("Restored"); axes[8].axis("off")
if save:
    plt.savefig(os.path.join(save_dir, "inferences.png"))

plt.show()