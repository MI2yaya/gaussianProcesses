from torch.utils.data import TensorDataset, DataLoader
from data.MNIST.mnistDatasetLoader import MnistDataloader
import sys, os
import torch
from copy import deepcopy
import numpy as np
from sklearn.metrics import mean_squared_error

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


'''
7.2 Constant Velocity Model
x = [px,vx,py,vy] #generalize into N targets
'''
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
dataSamples=100
min_dim=32

def fuse(xt,yt):
    return np.concatenate([xt,yt],axis=1)

def trainingData(dataSamples=10, dataTime=100, dt=1, r=1, q=1, trackers=1):
    data = []
    for _ in range(dataSamples):
        x,y = constantVelocityModel(trials=dataTime,dt=dt,r=r,q=q,trackers=trackers)
        fused = fuse(x,y)
        data.append(fused)
    return np.array(data)

def constantVelocityModel(trials=10, dt=1, r=1, q=1,trackers=1):
    x_initial = np.random.multivariate_normal(np.zeros(4*trackers), np.eye(4*trackers))
    xs = [x_initial]
    ys = [np.array([x_initial[i] for i in range(0, 4*trackers, 2)])]
    x = x_initial
    for _ in range(trials):
        w = np.random.multivariate_normal(np.zeros(4*trackers), q**2 * np.eye(4*trackers))
        A = np.eye(4*trackers)
        for i in range(0,trackers*4,2):
            A[i][i+1]=dt
        
        x = A @ x + w
        xs.append(x)
        y = np.random.multivariate_normal(np.zeros(2*trackers), r**2 * np.eye(2*trackers))
        
        H= np.zeros((2*trackers, 4*trackers))
        for i in range(0,trackers*2,1):
            H[i][2*i]=1
        
        y_observed = H @ x + y
        ys.append(y_observed)
    return xs, ys

measurementErrorsX = []
stateErrorsX = []
measurementErrorsY = []
stateErrorsY = []
xs_list = []
ys_list = []
MsX_list = []
MsY_list = []


for trial in range(kTrials):
    #1 generate data, KF
    xs, ys = constantVelocityModel(trials=dataTime,dt=dt,r=r_std,q=q_std,trackers=trackers)
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
    kf = KalmanFilter(x, P, F, H, Q, R)

    #2 Setup DDPM    
    data_array = trainingData(dataSamples=dataSamples, dataTime=dataTime, dt=dt, r=r_std, q=q_std, trackers=trackers)  # shape (dataSamples, time+1, 6)
    data_array = data_array.reshape(-1, data_array.shape[-1])  # (dataSamples*(time+1), 6)
    dataset = TensorDataset(torch.tensor(data_array, dtype=torch.float32))

    model = MLP(
        input_dim=data_array.shape[-1],
        hidden_dim=64,
        time_dim=32,
        num_res_blocks=2
    )
    diffusion = GaussianDiffusion(
        model, 
        timesteps=diffusion_timesteps,
        schedule="cosine",
        is_image_model=False
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
        ckpt_path=os.path.join('diffusionModels\data\CVM',"best_ema.pt"),
        is_image_model=False
    )
    
    trainer.train(steps=10000,log_every=500)
    
    filtered_states = []
    filtered_covs = []
    start_time = time.time()
    for t, y_obs in enumerate(ys):
        x_prior, _ = kf.predict()
        
        fused_vec = np.concatenate([x_prior, y_obs])
        fused_vec_norm = (fused_vec - fused_vec.mean()) / (fused_vec.std() + 1e-6)
        fused_tensor = torch.tensor(fused_vec_norm, dtype=torch.float32).unsqueeze(0)
        
        # Denoise using DDPM
        denoised = trainer.denoise(fused_tensor, timesteps=diffusion_timesteps)
        denoised_rescaled = denoised * fused_vec.std() + fused_vec.mean()
        denoised_flat = denoised_rescaled.view(-1).cpu().numpy()

        # Extract measurement part
        denoised_measurement = denoised_flat[-y_obs.shape[0]:]

        # Update KF with denoised measurement
        _, _ = kf.update(denoised_measurement)

        filtered_states.append(kf.x.copy())
        filtered_covs.append(kf.P.copy())
        if (t+1)%10==0:
            print(f"Trial {trial+1}/{kTrials}, Time step {t+1}/{len(ys)} processed in {time.time() - start_time:.4f} seconds")
            start_time = time.time()
        
    Ms = filtered_states
    MsX = [ele for ele in Ms]
    MsY = [H @ ele for ele in Ms]

    print(f"Msx: {MsX}")
    print(f"Msy: {MsY}")
    print(f"x: {xs}")
    print(f"y: {ys}")
    print("NaNs in xs:", np.isnan(xs).sum())
    print("NaNs in MsX:", np.isnan(MsX).sum())

    stateErrorsX.append(mean_squared_error([x[0] for x in xs], [m[0] for m in MsX]))
    stateErrorsY.append(mean_squared_error([x[2] for x in xs], [m[2] for m in MsX]))
    measurementErrorsX.append(mean_squared_error([y[0] for y in ys], [m[0] for m in MsY]))
    measurementErrorsY.append(mean_squared_error([y[1] for y in ys], [m[1] for m in MsY]))

    xs_list.append(xs)
    ys_list.append(ys)
    

if kTrials>0:
    for tracker in range(trackers):
        #X-pos plotting for tracker
        trackerX = [[x[0*tracker*4] for x in xs] for xs in xs_list]
        trackerY = [[y[0*tracker*2] for y in ys] for ys in ys_list]
        trackerMsX = [[m[0*tracker*4] for m in Ms] for Ms in MsX_list]
        trackerMsY = [[m[0*tracker*2] for m in Ms] for Ms in MsY_list]
        plotMSE(trackerX, trackerY, trackerMsX, trackerMsY, cov_ex=[cov[0][0] for cov in filtered_covs],r=r_std, q=q_std, save=save, name=f"_07p2a_tracker{tracker}_constant_velocity_model.png",title=f"C.V.M-X DnD Model Tracker #{tracker+1}/{trackers} Trials:{kTrials},time:{dataTime},r_std:{r_std},q_std:{q_std}")
        
        #Y-pos plotting for tracker
        trackerX = [[x[2+tracker*4] for x in xs] for xs in xs_list]
        trackerY = [[y[1+tracker*2] for y in ys] for ys in ys_list]
        trackerMsX = [[m[2+tracker*4] for m in Ms] for Ms in MsX_list]
        trackerMsY = [[m[1+tracker*2] for m in Ms] for Ms in MsY_list]
        plotMSE(trackerX, trackerY,trackerMsX, trackerMsY, r=r_std, q=q_std, cov_ex=[cov[2][2] for cov in filtered_covs], save=save, name=f"_07p2b_tracker{tracker}_constant_velocity_model.png",title=f"C.V.M-Y DnD Model Tracker #{tracker+1}/{trackers} Trials:{kTrials},time:{dataTime},r_std:{r_std},q_std:{q_std}")
        
        
        plotHist(stateErrorsX, measurementErrorsX, r_std, q_std, dataTime, kTrials, save=save, name=f"_07p2c_tracker{tracker}_constant_velocity_model.png",title=f"C.V.M X Errors Tracker #{tracker+1}/{trackers}")
        plotHist(stateErrorsY, measurementErrorsY, r_std, q_std, dataTime, kTrials, save=save, name=f"_07p2d_tracker{tracker}_constant_velocity_model.png",title=f"C.V.M Y Errors Tracker #{tracker+1}/{trackers}")
