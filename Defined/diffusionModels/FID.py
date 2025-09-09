import torch
import numpy as np
from torchvision.models import inception_v3
import torch.nn.functional as F
from scipy.linalg import sqrtm


def get_inception_model(device):
    """Load InceptionV3 truncated at the last pooling layer."""
    model = inception_v3(pretrained=True, transform_input=False)
    model.fc = torch.nn.Identity()  # remove final classifier
    model.eval()
    model.to(device)
    return model

def get_activations(dataloader, model, device, max_images=None):
    """Extract Inception activations for images in a dataloader."""
    activations = []
    count = 0
    for x, _ in dataloader:
        x = x.to(device)
        if x.shape[1] == 1:  # MNIST is grayscale, convert to RGB
            x = x.repeat(1, 3, 1, 1)
        x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        with torch.no_grad():
            pred = model(x)
        activations.append(pred.cpu().numpy())
        count += len(x)
        if max_images and count >= max_images:
            break
    return np.concatenate(activations, axis=0)

def calculate(mu1, sigma1, mu2, sigma2, eps=1e-6):
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid


#main
def compute_fid(real_dataloader, fake_images, device):
    model = get_inception_model(device)

    # Real activations
    real_acts = get_activations(real_dataloader, model, device, max_images=10000)
    mu_real, sigma_real = real_acts.mean(axis=0), np.cov(real_acts, rowvar=False)

    # Fake activations
    fake_ds = torch.utils.data.TensorDataset(fake_images, torch.zeros(len(fake_images)))
    fake_loader = torch.utils.data.DataLoader(fake_ds, batch_size=64)
    fake_acts = get_activations(fake_loader, model, device)
    mu_fake, sigma_fake = fake_acts.mean(axis=0), np.cov(fake_acts, rowvar=False)

    fid = calculate(mu_real, sigma_real, mu_fake, sigma_fake)
    return fid
