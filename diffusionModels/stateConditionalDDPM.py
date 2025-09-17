import torch
from torch.utils.data import TensorDataset, DataLoader
from data.MNIST.mnistDatasetLoader import MnistDataloader
import sys, os
import matplotlib.pyplot as plt
import math
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Defined.diffusionModels.Trainer import DiffusionTrainer
from Defined.diffusionModels.GaussianDiffusion import GaussianDiffusion
from Defined.diffusionModels.UNet import UNet
from Defined.diffusionModels.MLP import MLP

import torch.nn.functional as F
import random

random.seed(42)

save=False
save_dir = os.path.join("diffusionModels", "figs")
os.makedirs(save_dir, exist_ok=True)

def trainingData(dataSamples=10, dataTime=100, dt=1, r=1, q=1, trackers=1):
    noisy_vals=[]
    true_vals=[]
    for sample in range(dataSamples):
        noisy, true = constantVelocityModel(trials=dataTime, dt=dt, r=r, q=q, trackers=trackers)
        noisy_vals.append(noisy)
        true_vals.append(true)
    return np.array(noisy_vals),np.array(true_vals)

def constantVelocityModel(trials=10, dt=1, r=1, q=1,trackers=1):
    x_initial = np.random.multivariate_normal(np.zeros(4*trackers), np.eye(4*trackers))
    xs = [x_initial]
    ys = [x_initial]
    x = x_initial
    for _ in range(trials):
        w = np.random.multivariate_normal(np.zeros(4*trackers), q**2 * np.eye(4*trackers))
        A = np.eye(4*trackers)
        for i in range(0,trackers*4,2):
            A[i][i+1]=dt
        
        x = A @ x #true
        xs.append(x)
        
        y = x + w #noisy states
        ys.append(y)
    return ys, xs #flipped to return noisy, true

noisy_states,true_states = trainingData(dataSamples=2)

batch_size=1

dataset = TensorDataset(torch.tensor(noisy_states), 
                        torch.tensor(true_states))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("using cuda:", torch.cuda.is_available())
ckpt_path = os.path.join('diffusionModels','data','CVM','best_ema.pt')


preload = False
model = MLP(
    input_dim=4,
    hidden_dim=64,
    time_dim=32,
    num_res_blocks=2
).to(device)

diffusion = GaussianDiffusion(
    model,
    timesteps=1000,
    schedule="cosine",
    target='x0'
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
    use_EMA=True,
    ckpt_path=ckpt_path,
    is_image_model=False,
    target='x0'
)

if preload:
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    
else:
    trainer.train(
        steps=5000,
        log_every=100
    )


model.eval()


xs,ys = constantVelocityModel(trials=100)

denoised_xs=trainer.denoise(xs,timesteps=1000)

fig = plt.figure(figsize=(10,5))
axis = fig.add_subplot(111)
axis.plot(xs)
plt.show()