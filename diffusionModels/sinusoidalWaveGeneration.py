import numpy as np
import matplotlib.pyplot as plt
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Defined.diffusionModels.Trainer import DiffusionTrainer
from Defined.diffusionModels.GaussianDiffusion import GaussianDiffusion
from Defined.diffusionModels.MLP import MLP
from Defined.diffusionModels.UNet import UNet
from Defined.diffusionModels.S4 import S4Backbone
from Defined.KFs.EKF import ExtendedKalmanFilter
from torch.utils.data import TensorDataset
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using cuda:", torch.cuda.is_available())

dataSteps = 10
dataSamples = 5000
trainingSteps = 100000
diffusionTimesteps = 200
model_type='S4' #MLP, UNet, S4
ckpt_path=os.path.join('diffusionModels','data','SinusoidalWaves',f'DDPM_{model_type}.pt')
q=0.001
np.random.seed(42)
preload=True
dt=.05


dataRange = np.arange(0, dataSteps, dt)

def sinusoidalWaves(steps=100,dt=1,q=1):
    xs= []
    ys= []
    amplitude=np.random.uniform(1,5)
    frequency=np.random.uniform(1,4)
    phase=np.random.uniform(0,2*np.pi)
    for step in range(steps*(int(dt**-1))):
        w = np.random.normal(0, q)
        x = np.sin(frequency*step*dt + phase)*amplitude
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

noisy_states = (noisy_states - mean) / std
true_states = (true_states - mean) / std
print(f"Data mean: {mean}, std: {std}, shape {noisy_states.shape}")
if model_type=='MLP':
    dataset = TensorDataset(
        torch.tensor(noisy_states,dtype=torch.float32), # input shape (B,L)
        torch.tensor(true_states,dtype=torch.float32) #ignored in noise measurement
    )
    model = MLP(
        input_dim=noisy_states.shape[1],
        hidden_dim=128,
        time_dim=32,
        num_res_blocks=2
    ).to(device)
elif model_type=='UNet':
    dataset = TensorDataset(
        torch.tensor(noisy_states,dtype=torch.float32).unsqueeze(1), #input shape (B,1,L)
        torch.tensor(true_states,dtype=torch.float32).unsqueeze(1)#ignored in noise measurement
    )
    model = UNet(
        dim=1,
        in_ch=1,
        out_ch=1,
        base_dim=32,
        dim_mults=(1, 2, 4, 8),
        time_dim=32
    ).to(device)
elif model_type=='S4':
    dataset = TensorDataset(
        torch.tensor(noisy_states,dtype=torch.float32).unsqueeze(-1), #input shape (B,L,1)
        torch.tensor(true_states,dtype=torch.float32).unsqueeze(-1) #ignored in noise measurement
    )
    model = S4Backbone(
        d_input=1,
        d_output=1,
        d_model=128,
        n_layers=6,
        dropout=0.1,
        prenorm=False
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
    batch_size=32,
    lr=1e-4,
    device=device,
    ema_decay=0.995, 
    patience=5,
    ckpt_path=ckpt_path,
    is_image_model=False
)
if preload:
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
else:
    trainer.train(
        steps=trainingSteps,
        log_every=500
    )
    loss = trainer.get_loss_history()
    fig = plt.figure(figsize=(6, 4))
    plt.plot(loss)
    plt.show()

model.eval()

batch_size = 5
seq_len = noisy_states.shape[1]

# Initialize x with proper shape
if model_type=='MLP':
    x_init = torch.randn(1, seq_len).to(device)
elif model_type=='UNet':
    x_init = torch.randn(1, 1, seq_len).to(device)
elif model_type=='S4':
    x_init = torch.randn(1, seq_len, 1).to(device)

seqs = np.array(diffusion.sample_state(batch_size=batch_size, steps=seq_len, x_init=x_init))
seqs = seqs * std + mean  # Denormalize

all_seq = seqs.flatten()
mean = all_seq.mean()
std = all_seq.std()
print(f"Sampled data mean: {mean}, std: {std}")

fig = plt.figure(figsize=(10, 6))
plt.title(f'Generated Sinusoidal Wave {model_type}')
plt.axis('off')
axis = fig.add_subplot(111)
for seq in seqs:
    axis.plot(dataRange, seq, label='Generated Wave')
plt.show()

compare=True
if compare:
    state_dict = torch.load(os.path.join('diffusionModels','data','SinusoidalWaves',f'DDPM_MLP.pt'), map_location=device)
    model = MLP(
        input_dim=noisy_states.shape[1],
        hidden_dim=128,
        time_dim=32,
        num_res_blocks=2
    ).to(device)
    model.load_state_dict(state_dict)
    x_init=torch.randn(1,seq_len).to(device)
    seqsMLP = np.array(diffusion.sample_state(batch_size=batch_size, steps=seq_len, x_init=x_init))
    seqsMLP = seqsMLP * std + mean  

    state_dict = torch.load(os.path.join('diffusionModels','data','SinusoidalWaves',f'DDPM_UNET.pt'), map_location=device)
    model = UNet(
        dim=1,
        in_ch=1,
        out_ch=1,
        base_dim=32,
        dim_mults=(1, 2, 4, 8),
        time_dim=32
    ).to(device)
    model.load_state_dict(state_dict)
    x_init = torch.randn(1, 1, seq_len).to(device)
    seqsUNET = np.array(diffusion.sample_state(batch_size=batch_size, steps=seq_len, x_init=x_init))
    seqsUNET = seqsUNET * std + mean  

    state_dict = torch.load(os.path.join('diffusionModels','data','SinusoidalWaves',f'DDPM_S4.pt'), map_location=device)
    model = S4Backbone(
        d_input=1,
        d_output=1,
        d_model=128,
        n_layers=6,
        dropout=0.1,
        prenorm=False
    ).to(device)
    model.load_state_dict(state_dict)
    x_init = torch.randn(1, seq_len, 1).to(device)
    seqsS4 = np.array(diffusion.sample_state(batch_size=batch_size, steps=seq_len, x_init=x_init))
    seqsS4 = seqsS4 * std + mean  


    fig = plt.figure(figsize=(10, 6))
    plt.title(f'Generated Sinusoidal Waves')
    plt.axis('off')
    axis = fig.add_subplot(311)
    axis.title('MLP')
    for seq in seqsMLP:
        axis.plot(dataRange, seq)
    axis2 = fig.add_subplot(312)
    axis2.title('UNet')
    for seq in seqsUNET:
        axis2.plot(dataRange, seq)
    axis3 = fig.add_subplot(313)
    axis3.title("S4")
    for seq in seqsS4:
        axis3.plot(dataRange, seq)
    
    
    
    plt.show()