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
ckpt_path=os.path.join('diffusionModels','data','ScalarWalk','DDPMAdjustedEKF.pt')
q=5
r=5
np.random.seed(42)
preload=True

def RandomWalk(steps=100,dt=1,q=1,r=1):
    x = np.random.multivariate_normal(np.zeros(2), np.eye(2))
    xs= [x]
    ys= [x]
    for step in range(steps):
        w = np.random.multivariate_normal(np.zeros(2), q**2 * np.eye(2))
        x = x + w
        xs.append(x)

        v = np.random.multivariate_normal(np.zeros(2), r**2 * np.eye(2))
        y = x + v
        ys.append(y)
    return np.array(xs), np.array(ys)

def trainingData(dataSamples=10, dataTime=100, dt=1, r=1, q=1):
    true_states = []
    noisy_states = []
    for sample in range(dataSamples):
        true_state, noisy_state = RandomWalk(steps=dataTime, dt=dt, r=r, q=q)

        true_states.append(true_state)
        noisy_states.append(noisy_state)
    return np.array(noisy_states), np.array(true_states)

true_states, noisy_states = RandomWalk(steps=dataSteps,dt=1,q=5,r=5)

fig = plt.figure(figsize=(10, 6))
plt.title('2D Random Walk with Noisy Observations')
plt.axis('off')
axis1 = fig.add_subplot(121)
axis1.plot(true_states[:,0], label='True State X', color='g')
axis1.plot(noisy_states[:,0], label='Noisy Observations X', color='r')
axis1.legend()
axis1.set_xlabel('X Position')

axis2 = fig.add_subplot(122)
axis2.plot(true_states[:,1], label='True State Y', color='g')
axis2.plot(noisy_states[:,1], label='Noisy Observations Y', color='r')
axis2.legend()
axis2.set_xlabel('Y Position')

plt.show()

noisy_states, true_states = trainingData(dataSamples=dataSamples, dataTime=dataSteps, dt=1, r=r, q=q)
EKF_estimates=[]
for noisy_state in noisy_states:
    x0 = np.array([0, 0])  
    P0 = np.eye(2) * 1  
    Q = np.eye(2) * q**2 
    R = np.eye(2) * r**2  
    def F(x):
        return x  
    def H(x):
        return x 
    ekf = ExtendedKalmanFilter(x0, P0, F, H, Q, R)
    estimates, covars, uncertainties = ekf.batch_filter(noisy_state)
    EKF_estimates.append(estimates)
EKF_estimates = np.array(EKF_estimates)

pick = np.random.randint(0,EKF_estimates.shape[0]-1)
fig = plt.figure(figsize=(10, 6))
plt.title('EKF on 2D Random Walk with Noisy Observations')
plt.axis('off')
axis1 = fig.add_subplot(121)
axis1.plot(true_states[pick,:,0], label='True State X', color='g')
axis1.plot(noisy_states[pick,:,0], label='Noisy Observations X', color='r')
axis1.plot(EKF_estimates[pick,:,0], label='EKF Estimate X', color='b')
axis1.legend()
axis2=fig.add_subplot(122)
axis2.plot(true_states[pick,:,1], label='True State Y', color='g')
axis2.plot(noisy_states[pick,:,1], label='Noisy Observations Y', color='r')
axis2.plot(EKF_estimates[pick,:,1], label='EKF Estimate Y', color='b')
axis2.legend()
plt.show()

dataset = TensorDataset(
    torch.tensor(EKF_estimates,dtype=torch.float32).reshape(-1,2), #reshaped to (dataSamples*dataSteps, 2)
    torch.tensor(true_states,dtype=torch.float32).reshape(-1,2) #this will be ignored when training noise but pairs are necessary for my code.
)

model = MLP(
    input_dim=2,
    hidden_dim=128,
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