import torch
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from tqdm import trange, tqdm
import os
import copy
import numpy as np

class EMA:
    # Explonential Moving Averages to smooth out the weights of the model during training
    def __init__(self, model, beta=0.999):
        self.beta = beta
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad and param.dtype in (torch.float16, torch.float32, torch.float64):
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.beta).add_(param.data, alpha=1 - self.beta)
                
                
class DiffusionTrainer:
    def __init__(self, model, diffusion, dataset, batch_size=64, lr=2e-4, device=None,
                 ema_decay=0.995, val_ratio=0.05, num_workers=0, pin_memory=False, clip_grad=1.0,patience=10, ckpt_path="best_ema.pt",predefined=False, is_image_model=True):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.diffusion = diffusion.to(self.device)
        self.is_image_model = is_image_model

        n = len(dataset)
        n_val = max(1, int(n * val_ratio))
        n_train = n - n_val
        train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))

        self.train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                       drop_last=True, num_workers=num_workers, pin_memory=pin_memory)
        self.val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                                     drop_last=False, num_workers=num_workers, pin_memory=pin_memory)

        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.ema = EMA(self.model, beta=ema_decay)
        self.clip_grad = clip_grad
        self.patience=patience
        
        self.ckpt_path=ckpt_path
        self.predefined=predefined

    @torch.no_grad()
    def _evaluate(self):
        self.model.eval()
        total, count = 0.0, 0
        for (x, *_) in self.val_loader:
            x = x.to(self.device, non_blocking=True)
            
            total += self.diffusion(x).item()
            count += 1
        self.model.train()
        return total / max(1, count)

    def train(self, steps=10000, log_every=100):
        self.model.train()
        best_val = float("inf")
        bad = 0
        step = 0
        pbar = tqdm(total=steps, dynamic_ncols=True)
        patience = self.patience
        while step < steps:
            for (x, *_) in self.train_loader:
                x = x.to(self.device, non_blocking=True)

                loss = self.diffusion(x)
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                if self.clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                self.opt.step()
                self.ema.update(self.model)

                step += 1
                if step % log_every == 0:
                    pbar.set_postfix(loss=f"{loss.item():.4f}")
                    pbar.update(log_every)

                    val = self._evaluate()
                    print(f"step {step}: train loss {loss.item():.4f} \n val: {val} best val: {best_val} bad: {bad} / {patience}")
                    if val < best_val:
                        best_val = val
                        bad = 0
                        self.ema_model = copy.deepcopy(self.model)
                        torch.save(self.ema_model.state_dict(), self.ckpt_path)
                    
                    else:
                        bad += 1
                        if bad >= patience:
                            pbar.write(f"Early stopping at step {step} (best val: {best_val:.4f})")
                            pbar.close()
                            return
                if step >= steps:
                    break
        pbar.close()

    @torch.no_grad()
    def sample(self, num_samples=16, image_size=28, channels=1, use_ema=True):
        self.model.eval()

        # backup and optionally swap in EMA weights
        backup = None
        if use_ema and os.path.exists(self.ckpt_path):
            backup = {name: p.data.clone() for name, p in self.model.named_parameters()}
            for name, p in self.model.named_parameters():
                if name in self.ema.shadow:
                    p.data.copy_(self.ema.shadow[name])

        # sample with either ema or current weights
        if self.predefined:
            imgs = self.diffusion.sample(batch_size=num_samples)
        else:
            imgs = self.diffusion.sample(batch_size=num_samples, image_size=image_size, channels=channels)

        # restore original weights if we swapped EMA
        if backup is not None:
            for name, p in self.model.named_parameters():
                p.data.copy_(backup[name])

        return imgs
    
    @torch.no_grad()
    def denoise(self, noisy_input, timesteps=10):
        x = noisy_input.clone().to(next(self.model.parameters()).device)
        self.model.eval()

        # Pad/reshape for UNet if needed
        if self.is_image_model and x.ndim == 2:
            L = x.size(-1)
            H_dim = int(np.ceil(np.sqrt(L)))
            padded = torch.zeros(x.size(0), 1, H_dim, H_dim, device=x.device)
            padded.view(x.size(0), -1)[:, :L] = x
            x = padded

        # Denoise iteratively, this is kind of slow
        for t in reversed(range(timesteps)):
            t_tensor = torch.full((x.size(0),), t, device=x.device, dtype=torch.long)
            x = self.diffusion.p_sample(x, t_tensor)

        # Flatten if image
        if self.is_image_model and x.ndim > 2:
            x = x.view(x.size(0), -1)[:, :noisy_input.size(-1)]  # flatten & remove padding

        return x
