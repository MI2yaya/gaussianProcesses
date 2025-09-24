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

dataSteps = 1000
dataSamples= 100
trainingSteps=1000
diffusionTimesteps=100
ckpt_path=os.path.join('diffusionModels','data','SinusoidalWaves','DDPM.pt')
q=0.01
np.random.seed(42)
preload=False

def sinusoidalWaves(steps=100,dt=1,q=1):
    xs= [0]
    ys= [0]
    amplitude=np.random.uniform(0.5,2)
    frequency=np.random.uniform(0.1,0.5)
    phase=np.random.uniform(0,np.pi)
    for step in range(steps):
        w = np.random.normal(0, q)
        x = np.sin(2*np.pi*frequency*step*dt + phase)*amplitude
        xs.append(x)
        y = x + w
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

noisy_states, true_states = trainingData(dataSamples=1,dt=1,q=q)
fig = plt.figure(figsize=(10, 6))
plt.title('2D Random Walk with Noisy Observations')
plt.axis('off')
axis = fig.add_subplot(111)
axis.plot(true_states.reshape(-1,1), label='True State', color='g')
axis.plot(noisy_states.reshape(-1,1), label='Noisy Observations', color='r')
axis.legend()

plt.show()

noisy_states, true_states = trainingData(dataSamples=dataSamples, dataTime=dataSteps, dt=1, q=q)

dataset = TensorDataset(
    torch.tensor(noisy_states,dtype=torch.float32).reshape(-1,1), 
    torch.tensor(true_states,dtype=torch.float32).reshape(-1,1) #ignored during noise training
)

model = MLP(
    input_dim=2,
    hidden_dim=64,
    time_dim=32,
    num_res_blocks=2
)
diffusion = GaussianDiffusion(
    model,
    timesteps=diffusionTimesteps,
    schedule='cosine',
    is_image_model=False,
    target='noise'
)
trainer = DiffusionTrainer(
    model,
    diffusion,
    dataset=dataset,
    batch_size=32,
    lr=1e-3,
    device=device,
    ema_decay=0.995,
    patience=10,
    ckpt_path=ckpt_path,
    is_image_model=False,
    target='noise'
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

diffusion.sample_state()