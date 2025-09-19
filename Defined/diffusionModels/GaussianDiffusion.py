import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

def cosine_beta_schedule(timesteps: int, s: float = 0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-6, 0.999)


class GaussianDiffusion(nn.Module):
    def __init__(self, model, timesteps=1000, schedule="cosine",
                 beta_start=1e-4, beta_end=2e-2, is_image_model=True,target='noise'):
        super().__init__()
        
        self.model = model
        self.timesteps = timesteps
        self.is_image_model = is_image_model
        self.target = target #'noise or x0

        if schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            betas = torch.linspace(beta_start, beta_end, timesteps)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', 1.0 - betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(1.0 - betas, dim=0))
        self.register_buffer('alphas_cumprod_prev',
            torch.cat([torch.tensor([1.], device=betas.device), self.alphas_cumprod[:-1]], dim=0))

        # useful precomputes
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - self.alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / (1.0 - betas)))

        # posterior q(x_{t-1} | x_t, x_0) variance (Ho et al. 2020)
        posterior_variance = betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance.clamp(min=1e-20))
        self.register_buffer('posterior_log_variance', torch.log(self.posterior_variance))

        # x0 and prediction coeffs
        self.register_buffer('pred_x0_coeff', 1.0 / self.sqrt_alphas_cumprod)
        self.register_buffer('pred_eps_coeff', betas / self.sqrt_one_minus_alphas_cumprod)

    def _reshape(self, x, t):
        if self.is_image_model:
            return self.sqrt_alphas_cumprod[t].view(-1,1,1,1), self.sqrt_one_minus_alphas_cumprod[t].view(-1,1,1,1)
        else:
            return self.sqrt_alphas_cumprod[t].view(-1,1), self.sqrt_one_minus_alphas_cumprod[t].view(-1,1)

    def q_sample(self, x0, t, noise=None):
        #simulates 1 step of forward noising process
        if noise is None:
            noise = torch.randn_like(x0)
        s1, s2 = self._reshape(x0, t)
        return s1 * x0 + s2 * noise

    def forward(self, x, x0=None):
        #calculates loss, used during training
        b = x.size(0)
        device = x.device 
        t = torch.randint(0, self.timesteps, (b,), device=device, dtype=torch.long)

        noise = torch.randn_like(x) #generate noise
        x_noisy = self.q_sample(x, t, noise) #add noise to input
        pred_x = self.model(x_noisy, t) #predict noise or x0

        if self.target == 'noise':
            loss = F.mse_loss(pred_x, noise) #compare predicted noise to true noise, learns to denoise

        elif self.target == 'x0':
            if x0 is None:
                raise ValueError("x0 must be provided when targets='x0'")
            loss = F.mse_loss(pred_x, x0) #compare predicted x0 to true x0, learns true state

        else:
            raise ValueError(f"Unknown target type: {self.target}")

        return loss

    @torch.no_grad()
    def sample(self, batch_size, image_size=28, channels=1):
        #iteratively denoises from pure noise using learned model
        device = next(self.model.parameters()).device
        x = torch.randn(batch_size, channels, image_size, image_size, device=device)

        for t in reversed(range(self.timesteps)):
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x = self.p_sample(x, t_tensor)

        return x.clamp(-1, 1)
    
    @torch.no_grad()
    def p_sample(self, x, t):
        #single step of reverse denoising process
        if self.is_image_model:
            shape = (x.size(0), 1, 1, 1)
        else:
            shape = (x.size(0), 1)

        betas_t = self.betas[t].view(*shape)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(*shape)
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t].view(*shape)

        eps_theta = self.model(x, t)
        model_mean = sqrt_recip_alphas_t * (x - betas_t * eps_theta / sqrt_one_minus_alphas_cumprod_t)

        if (t > 0).any():
            posterior_var_t = self.posterior_variance[t].view(*shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_var_t) * noise
        else:
            return model_mean

    @torch.no_grad()
    def ddim_sample(self, batch_size=16, timesteps=5, eta=0.0, device=None, shape=None):
        device or next(self.model.parameters()).device
        shape = shape or (batch_size, self.channels, self.image_size, self.image_size)

        x = torch.randn(shape, device=device)

        # linear time steps for simplicity
        ts = torch.linspace(0, self.timesteps - 1, timesteps, device=device).long()
        alpha = self.alphas_cumprod  # precomputed alpha schedule
        sqrt_alpha = torch.sqrt(alpha)
        sqrt_one_minus_alpha = torch.sqrt(1 - alpha)

        for i in reversed(range(timesteps)):
            t = ts[i]
            eps = self.model(x, torch.full((x.size(0),), t, device=device, dtype=torch.long))
            x0_pred = (x - sqrt_one_minus_alpha[t] * eps) / sqrt_alpha[t]
            if i > 0:
                x = sqrt_alpha[ts[i - 1]] * x0_pred + sqrt_one_minus_alpha[ts[i - 1]] * eps
            else:
                x = x0_pred
        return x
    
    @torch.no_grad()
    def posterior_sample(self, y_meas, A, A_T, lam=.1, sigma_y=0.01):
        device = y_meas.device
        x = y_meas.clone()
        
        step_to_show = None
        
        for i, t in enumerate(reversed(range(self.timesteps))):
            t_tensor = torch.full((x.size(0),), t, device=device, dtype=torch.long)
            x = self.p_sample(x, t_tensor)
            
            diff = A(x) - y_meas
            grad = A_T(diff) / (sigma_y ** 2)
            x = x - lam * grad
            x = x.clamp(-1, 1)


            if i == step_to_show:
                plt.figure(figsize=(3,3))
                plt.imshow(x[0,0].cpu().clamp(-1,1), cmap='gray')
                plt.title(f"Timestep {t}")
                plt.axis('off')
                plt.show()

        return x
    
    
    @torch.no_grad()
    def denoise_states(self,noisy_states):
        device = noisy_states.device
        x = noisy_states.clone()

        for t in reversed(range(self.timesteps)):
            t_tensor = torch.full((x.size(0),), t, device=device, dtype=torch.long)
            x = self.p_sample(x, t_tensor)  # already works for non-image since _reshape handles it

        return x