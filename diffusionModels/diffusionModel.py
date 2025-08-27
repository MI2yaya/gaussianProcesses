import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion

model = Unet(
    dim = 64,
    dim_mults= (1,2,4,8)
)

diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 100
)

training_images = torch.randn(8,3,128,128)
loss = diffusion(training_images)
loss.backward()

sampled_images = diffusion.sample(batch_size = 4)
print(sampled_images.shape)  # (4, 3, 128, 128