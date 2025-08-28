from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import torch
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
from os.path import join
from torchvision import datasets, transforms
from data.MNIST.mnistDatasetLoader import MnistDataloader
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Defined.diffusionModels.diffusionTrainer import SimpleTrainer


mnist_dataloader = MnistDataloader()
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()
print(x_train.shape)
print(y_train.shape)

x_train = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1) / 255. * 2 - 1
y_train = torch.tensor(y_train, dtype=torch.long)

dataset = TensorDataset(x_train, y_train)

# Define model and diffusion
model = Unet(
    dim=64, 
    dim_mults=(1, 2, 4),
    channels=1
)

diffusion = GaussianDiffusion(model, image_size=28, timesteps=10)

# Trainer
trainer = SimpleTrainer(model, diffusion, dataset, batch_size=2, lr=2e-5)

trainer.train(steps=1000)

sampled_images = trainer.sample(num_samples=16)

# Plot
fig = plt.figure(figsize=(6, 6))
axes = fig.subplots(4, 4)
for i, ax in enumerate(axes.flat):
    img = sampled_images[i].detach().cpu().numpy().transpose(1, 2, 0)
    img = (img + 1) / 2
    ax.imshow(img.squeeze(), cmap="gray")
    ax.axis("off")

plt.show()
