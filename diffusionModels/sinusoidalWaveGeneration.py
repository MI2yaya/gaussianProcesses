import numpy as np
import matplotlib.pyplot as plt
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Defined.diffusionModels.Trainer import DiffusionTrainer
from Defined.diffusionModels.GaussianDiffusion import GaussianDiffusion
from Defined.diffusionModels.MLP import MLP
from Defined.KFs.EKF import ExtendedKalmanFilter
from torch.utils.data import TensorDataset
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using cuda:", torch.cuda.is_available())

dataSteps = 20
dataSamples= 5000
trainingSteps= 100000
diffusionTimesteps=1000
ckpt_path=os.path.join('diffusionModels','data','SinusoidalWaves','DDPM.pt')
q=0.001
np.random.seed(42)
preload=False
dt=.05

dataRange = np.arange(0, dataSteps, dt)

def sinusoidalWaves(steps=100,dt=1,q=1):
    xs= []
    ys= []
    amplitude=np.random.uniform(1,10)
    frequency=np.random.uniform(.5,4)
    phase=np.random.uniform(0,2*np.pi)
    for step in range(steps*(int(dt**-1))):
        w = np.random.normal(0, q)
        x = np.sin(frequency*step*dt + phase)*amplitude
        xs.append(x)
        y = x + w*amplitude
        ys.append(y)
    return xs, ys

def trainingData(dataSamples=10, dataTime=100, dt=1,q=1):
    true_states = []
    noisy_states = []
    for sample in range(dataSamples):
        true_state, noisy_state = sinusoidalWaves(steps=dataTime, dt=dt,q=q)

        true_states.append(true_state)
        noisy_states.append(noisy_state)
    return np.array(noisy_states), np.array(true_states)

noisy_states, true_states = trainingData(dataSamples=10,dataTime=dataSteps,dt=dt,q=q)
fig = plt.figure(figsize=(10, 6))
plt.title('Sinusoidal Wave with Noisy Observations')
plt.axis('off')
axis = fig.add_subplot(111)

for true_state, noisy_state in zip(true_states, noisy_states):
    axis.plot(dataRange,true_state, label='True State')
    axis.legend()

plt.show()

noisy_states, true_states = trainingData(dataSamples=dataSamples, dataTime=dataSteps, dt=dt, q=q)
# Flatten over samples and timesteps
all_noisy = noisy_states.flatten()
all_true = true_states.flatten()

mean = all_noisy.mean()
std = all_noisy.std()

noisy_states_norm = (noisy_states - mean) / std
true_states_norm = (true_states - mean) / std
print(f"Data mean: {mean}, std: {std}, shape {noisy_states.shape}")
dataset = TensorDataset(
    torch.tensor(noisy_states_norm,dtype=torch.float32),
    torch.tensor(true_states_norm,dtype=torch.float32)
)

model = MLP(
    input_dim=noisy_states.shape[1],
    hidden_dim=512,
    time_dim=32,
    num_res_blocks=6
).to(device)

diffusion = GaussianDiffusion(
    model,
    timesteps=diffusionTimesteps,
    schedule='cosine',
    is_image_model=False,
    target='x0'
).to(device)
trainer = DiffusionTrainer(
    model,
    diffusion,
    dataset=dataset,
    batch_size=128,
    lr=1e-4,
    device=device,
    ema_decay=0.995, 
    patience=20,
    ckpt_path=ckpt_path,
    is_image_model=False
)
if preload:
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
else:
    trainer.train(
        steps=trainingSteps,
        log_every=100
    )
    loss = trainer.get_loss_history()
    fig = plt.figure(figsize=(6, 4))
    plt.plot(loss)
    plt.show()

model.eval()

seq_norms = np.array(diffusion.sample_state(batch_size=5,steps=noisy_states_norm.shape[1]))
seqs = seq_norms * std + mean  # Denormalize

all_seq = seqs.flatten()
mean = all_seq.mean()
std = all_seq.std()
print(f"Sampled data mean: {mean}, std: {std}")

fig = plt.figure(figsize=(10, 6))
plt.title('Generated Sinusoidal Wave')
plt.axis('off')
axis = fig.add_subplot(111)
for seq in seqs:
    axis.plot(dataRange, seq, label='Generated Wave')
plt.show()