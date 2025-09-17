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

# Load MNIST
mnist_dataloader = MnistDataloader()
(x_train, y_train), _ = mnist_dataloader.load_data()
x_train = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1) / 255.0 * 2 - 1
y_train = torch.tensor(y_train, dtype=torch.long)

batch_size=32

dataset = TensorDataset(x_train, y_train)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("using cuda:", torch.cuda.is_available())

timestepTesting=[25,100,200]
trainingStepTesting=[1000,1000,1000]
save=True
num_trials=5
fid_samples=50

results = {t: {"losses": [], "fids": []} for t in timestepTesting}
for k, timestep in enumerate(timestepTesting):
    trainingStep = trainingStepTesting[k]

    for trial in range(num_trials):
        print(f"\n=== Trial {trial+1}/{num_trials}, Timesteps={timestep} ===")

        model = UNet(
            in_channels=1,
            out_channels=1,
            base_dim=64,
            dim_mults=(1, 2, 4)
        ).to(device)

        diffusion = GaussianDiffusion(
            model,
            timesteps=timestep,
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
            use_EMA=False
        )

        trainer.train(
            steps=trainingStep,
            log_every=100,
            fid_every=trainingStep,
            fid_samples=fid_samples
        )

        # Save histories
        results[timestep]["losses"].append(trainer.get_loss_history())
        results[timestep]["fids"].append(trainer.get_fid_history())

        print("LAST FIDs:", results[timestep]["fids"][-1][-1][1] if results[timestep]["fids"][-1] else "No FID")


save_dir = os.path.join("diffusionModels", "figs")
os.makedirs(save_dir, exist_ok=True)

fig, ax = plt.subplots(figsize=(7, 6))

for timestep in timestepTesting:
    all_losses = results[timestep]["losses"]

    max_len = max(len(l) for l in all_losses)
    losses_matrix = np.full((len(all_losses), max_len), np.nan)

    for i, trial_losses in enumerate(all_losses):
        losses_matrix[i, :len(trial_losses)] = trial_losses

    mean_loss = np.nanmean(losses_matrix, axis=0)
    std_loss = np.nanstd(losses_matrix, axis=0)

    steps = np.arange(1, max_len+1)

    ax.plot(steps, mean_loss, label=f"Timesteps={timestep}")
    ax.fill_between(steps, mean_loss - std_loss, mean_loss + std_loss, alpha=0.2)

ax.set_yscale("log")
ax.set_xlabel("Training Steps")
ax.set_ylabel("Loss")
ax.set_title(f"Training Loss (mean ± std, {num_trials} trials)")
ax.legend()
plt.tight_layout()
if save:
    plt.savefig(os.path.join(save_dir, "Losses.png"))
plt.show()


fig, ax = plt.subplots(figsize=(7, 6))

colors = plt.cm.tab10.colors
legend_handles = []

for timestep_idx, timestep in enumerate(timestepTesting):
    all_fids = results[timestep]["fids"]

    color = colors[timestep_idx % len(colors)]
    for trial_idx, trial_fids in enumerate(all_fids):
        if not trial_fids:
            continue
        steps, vals = zip(*trial_fids)
        ax.scatter(steps, vals, color=color, alpha=0.7)
        ax.plot(steps, vals, color=color, alpha=0.3) 

    handle = plt.Line2D([0], [0], marker='o', color=color, label=f"Timesteps={timestep}", linestyle='')
    legend_handles.append(handle)

ax.set_xlabel("Training Step")
ax.set_ylabel("FID Score")
ax.set_title(f"FID Scores (fid_samples={fid_samples}, {num_trials} trials)")
ax.legend(handles=legend_handles)
plt.tight_layout()
if save:
    plt.savefig(os.path.join(save_dir, "FIDScores.png"))
plt.show()