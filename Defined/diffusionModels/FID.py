import torch
import numpy as np
from torchvision.models import inception_v3
import torch.nn.functional as F
from scipy.linalg import sqrtm

def get_inception_model(device):
    model = inception_v3(weights="IMAGENET1K_V1", transform_input=False)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    return model

@torch.no_grad()
def get_activations(dataloader, model, device, max_images=None):
    activations = []
    count = 0
    for x, _ in dataloader:
        x = x.to(device)
        if x.shape[1] == 1:  # grayscale → RGB
            x = x.repeat(1, 3, 1, 1)
        x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)

        with torch.cuda.amp.autocast():  # speedup
            pred = model(x)

        activations.append(pred.cpu().numpy())
        count += len(x)
        if max_images and count >= max_images:
            break
    return np.concatenate(activations, axis=0)

def calculate_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)

def compute_fid(real_stats, fake_images, model, device, batch_size=64):
    mu_real, sigma_real = real_stats

    fake_ds = torch.utils.data.TensorDataset(fake_images, torch.zeros(len(fake_images)))
    fake_loader = torch.utils.data.DataLoader(fake_ds, batch_size=batch_size)

    fake_acts = get_activations(fake_loader, model, device)
    mu_fake, sigma_fake = fake_acts.mean(axis=0), np.cov(fake_acts, rowvar=False)

    return calculate_fid(mu_real, sigma_real, mu_fake, sigma_fake)