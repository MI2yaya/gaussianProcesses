import torch
from torch.utils.data import TensorDataset, DataLoader
from data.MNIST.mnistDatasetLoader import MnistDataloader
import sys, os
import matplotlib.pyplot as plt
from PIL import Image
import math

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

timestepTesting=[25,100,200,1000,10000]
trainingStepTesting=[10000,10000,10000,10000,10000]
save=True
losses=[]
fids=[]
for k,timestep in enumerate(timestepTesting):
    trainingStep=trainingStepTesting[k]
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
        #ckpt_path=os.path.join('diffusionModels\data\MNIST',"best_ema.pt"),
        use_EMA=False
    )

    print("Starting training...")
    trainer.train(steps=trainingStep, log_every=100,fid_every=trainingStep,fid_samples=500)
    losses.append(trainer.get_loss_history())

    fid = trainer.get_fid_history()
    print("LAST FIDs:", fid[-1][1] if fid else "No FID calculated")
    fids.append(fid)

    print("Sampling images...")
    sampled_images = trainer.sample(num_samples=16)
    # Plot
    fig, axes = plt.subplots(4, 4, figsize=(6,6))
    fig.suptitle(f"Sampled Images after Trial {k+1}, Timesteps={timestep}, Training Steps={trainingStep}")
    for i, ax in enumerate(axes.flat):
        img = sampled_images[i].detach().cpu().numpy().transpose(1, 2, 0)
        img = (img + 1) / 2  # rescale 0-1
        ax.imshow(img.squeeze(), cmap="gray")
        ax.axis("off")

    if save:
        plt.savefig(os.path.join("diffusionModels","figs",f"sampledImagesTrial{k+1}_timesteps{timestep}_steps{trainingStep}.png"))

    #plt.show()
    plt.close()

if save:
    save_dir = os.path.join("diffusionModels", "figs")
    imgs = []

    # Collect images
    for img_path in sorted(os.listdir(save_dir)):
        if img_path.startswith("sampledImagesTrial"):
            img = Image.open(os.path.join(save_dir, img_path))
            imgs.append(img.copy())  # keep a copy
            img.close()
            os.remove(os.path.join(save_dir, img_path))

    if imgs:
        total_width = sum(img.width for img in imgs)
        max_height = max(img.height for img in imgs)
        combined = Image.new('RGB', (total_width, max_height))

        x_offset = 0
        for img in imgs:
            combined.paste(img, (x_offset, 0))
            x_offset += img.width

        combined.save(os.path.join(save_dir, "all_sampled_images.png"))


for i in range(len(losses)):
    if len(losses[i])<max(trainingStepTesting):
        losses[i]+=[losses[i][-1]]*(max(trainingStepTesting)-len(losses[i]))
    

fig,ax=plt.subplots(1,1,figsize=(6,6))
for k in range(len(timestepTesting)):
    ax.plot(losses[k],label=f'Timesteps={timestepTesting[k]}, Training Steps={trainingStepTesting[k]}')
    ax.axvline(x=trainingStepTesting[k], color='gray', linestyle='--', linewidth=0.8)

ax.legend()
ax.set_yscale('log')
ax.set_xlabel('Training Steps')
ax.set_ylabel('Loss')
if save:
    plt.savefig(os.path.join("diffusionModels","figs",f"losses.png"))
plt.show()


fig, ax = plt.subplots(1, 1, figsize=(6, 6))

for k in range(len(timestepTesting)):
    fid_steps, fid_values = zip(*fids[k])
    ax.plot(fid_steps, fid_values, marker='o', label=f'Timesteps={timestepTesting[k]}, Steps={trainingStepTesting[k]}')

ax.set_xlabel("Training Step")
ax.set_ylabel("FID Score")
ax.set_title("FID Progression Across Trials")
ax.legend()
if save:
    plt.savefig(os.path.join("diffusionModels","figs",f"FIDScores.png"))
plt.show()