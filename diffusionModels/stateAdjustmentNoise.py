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
torch.manual_seed(42)
kTrials=1
dataTime=100
r_std=5
q_std=5
dt=1
trackers=1
diffusion_timesteps=1000
batch_size=32
dataSamples=10000
min_dim=32
target='noise' #noise or x0
preload=True
ckpt_path=os.path.join('diffusionModels\data\CVM',"best_ema_noise.pt")


def trainingData(dataSamples=10, dataTime=100, dt=1, r=1, q=1, trackers=1):
    true_states = []
    noisy_states = []
    for sample in range(dataSamples):
        true_state, noisy_state, _ = constantVelocityModel(trials=dataTime, dt=dt, r=r, q=q, trackers=trackers)

        true_states.append(true_state)
        noisy_states.append(noisy_state)
    return np.array(noisy_states), np.array(true_states)

def constantVelocityModel(trials=10, dt=1, r=1, q=1,trackers=1):
    x_initial = np.random.multivariate_normal(np.zeros(4*trackers), np.eye(4*trackers))
    noisy_state = [x_initial]
    true_state = [x_initial]
    measurements= [np.array([x_initial[i] for i in range(0, 4*trackers, 2)])]
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
        
        z = np.random.multivariate_normal(np.zeros(2*trackers), r**2 * np.eye(2*trackers))
        
        H= np.zeros((2*trackers, 4*trackers))
        for i in range(0,trackers*2,1):
            H[i][2*i]=1
        
        measurement = H @ x + z
        measurements.append(measurement)
    return true_state,noisy_state, measurements

measurementErrorsX = []
stateErrorsX = []
measurementErrorsY = []
stateErrorsY = []
xs_list = []
ys_list = []
MsX_list = []
MsY_list = []
adjustedMsX_list = []


noisy_states, true_states = trainingData(dataSamples=dataSamples, dataTime=dataTime, dt=dt, r=r_std, q=q_std, trackers=trackers)  

dataset = TensorDataset(
    torch.tensor(noisy_states, dtype=torch.float32).reshape(-1, noisy_states.shape[-1]),
    torch.tensor(true_states, dtype=torch.float32).reshape(-1, true_states.shape[-1]) #this will be ignored if target='noise'
)

model = MLP(
    input_dim=noisy_states.shape[-1],
    hidden_dim=128,
    time_dim=32,
    num_res_blocks=3
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
        steps=10000,
        log_every=100
    )
    loss = trainer.get_loss_history()
    fig = plt.figure(figsize=(6, 4))
    plt.plot(loss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
    


model.eval()

true,noisy,measurements = constantVelocityModel(trials=dataTime, dt=dt, r=r_std, q=q_std, trackers=trackers)
x = np.zeros(4*trackers)  
P = np.eye(4*trackers)

F = np.eye(4*trackers)
for i in range(0, trackers*4, 2):
    F[i][i+1] = dt

H = np.zeros((2*trackers, 4*trackers))
for i in range(0, trackers*2, 1):
    H[i][2*i] = 1

R = np.eye(2*trackers) * r_std**2
Q = np.eye(4*trackers) * q_std**2 
kfAdjusted = KalmanFilter(x, P, F, H, Q, R)
kfNormal = KalmanFilter(x, P, F, H, Q, R)
xs=[]

for z in measurements:
    x_pred, P_pred = kfAdjusted.predict()
    x_denoised = diffusion.denoise_states(torch.tensor(x_pred,dtype=torch.float32, device=device).unsqueeze(0)).squeeze(0).cpu().numpy()
    print(f"\nx_pred:{x_pred}\nx_denoised:{x_denoised},\nz:{z}")
    delta = x_denoised - x_pred
    max_delta=10
    norm_delta = np.clip(delta, -max_delta, max_delta)
    x_denoised = x_pred + norm_delta
    kfAdjusted.x = x_denoised

    kfAdjusted.update(z)
    xs.append(kfAdjusted.x)

Ms, Covs= kfNormal.batch_filter(measurements)

fig = plt.figure(figsize=(12, 6))
axisX = fig.add_subplot(121)
axisX.plot([state[0] for state in true], label='True Position X', color='g')
axisX.plot([state[0] for state in noisy], label='Noisy Position X', color='r', alpha=0.5)
axisX.plot([state[0] for state in measurements], label='Measurements X', color='orange', alpha=0.5)
axisX.plot([state[0] for state in xs], label='Kalman Filter Adjusted X', color='b')
axisX.plot([state[0] for state in Ms], label='Kalman Filter Normal X', color='purple', linestyle='--')
axisX.legend()

axisY = fig.add_subplot(122)
axisY.plot([state[2] for state in true], label='True Position Y', color='g')
axisY.plot([state[2] for state in noisy], label='Noisy Position Y', color='r', alpha=0.5)
axisY.plot([state[1] for state in measurements], label='Measurements Y', color='orange', alpha=0.5)
axisY.plot([state[2] for state in xs], label='Kalman Filter Adjusted Y', color='b')
axisY.plot([state[2] for state in Ms], label='Kalman Filter Normal Y', color='purple', linestyle='--')
axisY.legend()
plt.show()

print("Position X MSE Adjusted:", mean_squared_error([state[0] for state in true], [state[0] for state in xs]))
print("Position X MSE Normal:", mean_squared_error([state[0] for state in true], [state[0] for state in Ms]))
print("Position Y MSE Adjusted:", mean_squared_error([state[2] for state in true], [state[2] for state in xs]))
print("Position Y MSE Normal:", mean_squared_error([state[2] for state in true], [state[2] for state in Ms]))