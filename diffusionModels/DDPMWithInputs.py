import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np


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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=1, out_channels=1, base_dim=64, dim_mults=(1,2,4)).to(device)
ckpt_path = os.path.join('diffusionModels','data','MNIST','best_ema_inference.pt')
ckpt_path = os.path.join('diffusionModels','data','MNIST','best_ema_inference.pt')
state_dict = torch.load(ckpt_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()

diffusion = GaussianDiffusion(model, timesteps=1000, schedule="cosine").to(device)


img_path = os.path.join('diffusionModels','data','inputImages','')
img = Image.open(img_path).convert("L")
img = img.resize((28,28))
img = np.array(img).astype(np.float32) / 255.0 
img = torch.tensor(img).unsqueeze(0).unsqueeze(0) * 2 - 1 
img = img.to(device)

A = lambda x: x
A_T = lambda x: x

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

kernel = get_gaussian_kernel(kernel_size=5, sigma=1.0, device=device)
kernel = kernel / kernel.sum()
y_meas = A(img, kernel)

# 4️⃣ Run posterior sampling
restored = diffusion.posterior_sample(
    y_meas=y_meas,
    A=lambda z: A(z, kernel),
    A_T=lambda z: A_T(z, kernel),
    lam=.1,
    sigma_y=1  # adjust to approximate measurement noise
)

restored_img = (restored.clamp(-1,1) + 1) / 2.0  # [-1,1] -> [0,1]
restored_img = restored_img.squeeze().cpu().numpy()

import matplotlib.pyplot as plt
# Convert tensors to [0,1] range for display
input_img = (img.squeeze().cpu().numpy() + 1) / 2
blurred_img = (y_meas.squeeze().cpu().numpy() + 1) / 2
restored_img = (restored.clamp(-1,1).squeeze().cpu().numpy() + 1) / 2

fig, axes = plt.subplots(1, 3, figsize=(9, 3))

axes[0].imshow(input_img, cmap="gray")
axes[0].set_title("Input")
axes[0].axis("off")

axes[1].imshow(blurred_img, cmap="gray")
axes[1].set_title("Blurred")
axes[1].axis("off")

axes[2].imshow(restored_img, cmap="gray")
axes[2].set_title("Restored")
axes[2].axis("off")

plt.show()
