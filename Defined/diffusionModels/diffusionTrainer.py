import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import trange, tqdm


class SimpleTrainer:
    def __init__(self, model, diffusion, dataset, batch_size=32, lr=2e-5):
        self.model = model
        self.diffusion = diffusion
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def train(self, steps=10000, log_every=100):
        step = 0
        self.model.train()
        pbar = tqdm(total=steps)
        while step < steps:
            for batch in self.dataloader:
                x = batch[0]
                loss = self.diffusion(x)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                step += 1
                if step % log_every == 0:
                    pbar.set_postfix({"loss": loss.item()})
                pbar.update(1)

                if step >= steps:
                    break
        pbar.close()

    @torch.no_grad()
    def sample(self, num_samples=16):
        self.model.eval()
        return self.diffusion.sample(batch_size=num_samples).cpu()
