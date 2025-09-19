from torch.utils.data import TensorDataset, DataLoader
from data.MNIST.mnistDatasetLoader import MnistDataloader
import sys, os
import torch
from copy import deepcopy
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Defined.diffusionModels.Trainer import DiffusionTrainer
from Defined.diffusionModels.GaussianDiffusion import GaussianDiffusion
from Defined.diffusionModels.UNet import UNet
from Defined.diffusionModels.MLP import MLP
from Defined.KFs.KF import KalmanFilter
from Defined.Helpers.plotting import plotMSE, plotHist 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using cuda:", torch.cuda.is_available())

np.random.seed(42)
kTrials=1
dataTime=100
r_std=5
q_std=5
dt=1
trackers=1
save=False
diffusion_timesteps=100
batch_size=32
dataSamples=10000
min_dim=32
target='x0' #noise or x0
preload=False
ckpt_path=os.path.join('diffusionModels\data\CVM',"best_ema.pt")


def trainingData(dataSamples=10, dataTime=100, dt=1, r=1, q=1, trackers=1):
    true_states = []
    noisy_states = []
    for sample in range(dataSamples):
        true_state, noisy_state = constantVelocityModel(trials=dataTime, dt=dt, r=r, q=q, trackers=trackers)

        true_states.append(true_state)
        noisy_states.append(noisy_state)
    return np.array(noisy_state), np.array(true_state)

def constantVelocityModel(trials=10, dt=1, r=1, q=1,trackers=1):
    x_initial = np.random.multivariate_normal(np.zeros(4*trackers), np.eye(4*trackers))
    noisy_state = [x_initial]
    true_state = [x_initial]
    x = x_initial
    for _ in range(trials):
        w = np.random.multivariate_normal(np.zeros(4*trackers), q**2 * np.eye(4*trackers))
        A = np.eye(4*trackers)
        for i in range(0,trackers*4,2):
            A[i][i+1]=dt
        
        x = A @ x
        x_noisy = x + w
        noisy_state.append(x_noisy)
        true_state.append(x)
    return true_state,noisy_state

measurementErrorsX = []
stateErrorsX = []
measurementErrorsY = []
stateErrorsY = []
xs_list = []
ys_list = []
MsX_list = []
MsY_list = []
adjustedMsX_list = []


noisy_states, true_states = trainingData(dataSamples=dataSamples, dataTime=dataTime, dt=dt, r=r_std, q=q_std, trackers=trackers)  # shape (dataSamples, time+1, 6)

dataset = TensorDataset(
    torch.tensor(noisy_states, dtype=torch.float32),
    torch.tensor(true_states, dtype=torch.float32)
)

model = MLP(
    input_dim=noisy_states.shape[-1],
    hidden_dim=64,
    time_dim=32,
    num_res_blocks=2
)
diffusion = GaussianDiffusion(
    model, 
    timesteps=diffusion_timesteps,
    schedule="cosine",
    is_image_model=False,
    target=target
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
    ckpt_path=ckpt_path,
    is_image_model=False,
    target=target
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




ex_noisy_states, ex_true_states = trainingData(
    dataSamples=1, dataTime=dataTime, dt=dt, r=r_std, q=q_std, trackers=trackers
)
ex_noisy_states = torch.tensor(ex_noisy_states, dtype=torch.float32).to(device)
ex_true_states  = torch.tensor(ex_true_states, dtype=torch.float32).to(device)

restored_states = diffusion.denoise_states(ex_noisy_states)

restored_states2 = diffusion.posterior_sample(
    y_meas=ex_noisy_states,
    A=lambda x: x,
    A_T=lambda x: x,
    lam=0.1,
    sigma_y=r_std
)
restored_states2 = restored_states2.clamp(-5, 5)

mse = mean_squared_error(
    ex_noisy_states.cpu().numpy().reshape(-1, ex_noisy_states.shape[-1]),
    ex_true_states.cpu().numpy().reshape(-1, ex_true_states.shape[-1])
)
print("Original MSE:", mse)
mse = mean_squared_error(
    ex_true_states.cpu().numpy().reshape(-1, ex_true_states.shape[-1]),
    restored_states.cpu().numpy().reshape(-1, restored_states.shape[-1])
)
print("Restored MSE:", mse)
mse = mean_squared_error(
    ex_true_states.cpu().numpy().reshape(-1, ex_true_states.shape[-1]),
    restored_states2.cpu().numpy().reshape(-1, restored_states2.shape[-1])
)
print("Restored MSE method 2:", mse)


plt.plot(ex_true_states[:,0], label="True X")
plt.plot(ex_noisy_states[:,0], label="Noisy X", alpha=0.5)
plt.plot(restored_states[:,0], label="Restored X")
plt.plot(restored_states2[:,0], label='Restored X-2')
plt.legend()
plt.show()