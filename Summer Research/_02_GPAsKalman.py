import numpy as np
import random
import matplotlib.pyplot as plt
from collections import namedtuple
from numpy.random import randn
import math
from filterpy.kalman import predict, update, KalmanFilter
from filterpy.common import Q_discrete_white_noise
from sklearn.metrics import mean_squared_error
from numpy.random import randn, seed
import time


def compute_dog_data(z_var, process_var, count=1, dt=.1):
    "returns track, measurements 1D ndarrays"
    x, vel = 0., 1.
    z_std = math.sqrt(z_var) 
    p_std = math.sqrt(process_var)
    xs, zs = [], []
    for _ in range(count):
        v = vel
        x += v*dt        
        xs.append(x)
        zs.append(x + randn() * z_std)        
    return np.array(xs), np.array(zs)

def gp_kernal(P,A,Q,C,N):
    K = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            # Initial state contribution
            Ai = np.linalg.matrix_power(A, i)
            Aj = np.linalg.matrix_power(A, j)
            term1 = C @ Ai @ P @ Aj.T @ C.T
            sum_term = 0.0
            for k in range(min(i, j)):
                Aik = np.linalg.matrix_power(A, i - 1 - k)
                Ajk = np.linalg.matrix_power(A, j - 1 - k)
                sum_term += (C @ Aik @ Q @ Ajk.T @ C.T)[0, 0]
            K[i, j] = term1[0, 0] + sum_term

    return K

def makeKF(x,P,F,Q,H,R):
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = x
    kf.F = F
    kf.H = H
    kf.R *= R
    kf.P = P
    kf.Q = Q  
    return kf

dt=.1 #time scale
N=100
#initial belief
x = np.array([0, 0])    #state: [pos,vel]
P = np.diag([500.,49])      #state covar
P0 = np.diag([500.,49]) 

#state
F = np.array([[1,dt],[0,1]]) #state transition matrix
Q = np.array([[0.01, 0.],
              [0., 0.01]])   #process noise covar

#measurement
H = np.array([[1., 0.]])     #state to measurement
R = np.array([[1]])           #measurement noise covar

track, zs = compute_dog_data(R[0, 0], Q[0, 0], count=N,dt=dt) #data


kf = makeKF(x,P,F,Q,H,R)
xs, cov, _, _ = kf.batch_filter(zs)
xs, cov, _, _ = kf.rts_smoother(xs, cov)

# Build the GP kernel from Kalman structure
K_clean = gp_kernal(P0, F, Q, H, N)
K_obs = K_clean + R[0, 0] * np.eye(N)
mu_post = K_clean @ np.linalg.inv(K_obs) @ zs


plt.plot(track, label='True Position')
plt.plot(zs, label='Measurements', linestyle='dotted')
plt.plot(xs[:, 0], label='Kalman Estimate')
plt.plot(mu_post, label='GP Estimate (Kalman Kernel)', linestyle='dashdot')
plt.legend()
plt.xlabel('Time Step')
plt.ylabel('Position')
plt.title('Kalman Filter on Dog Position Tracking')
plt.grid()
plt.show()







####DATAAAAA###########

random.seed(5)
seed(5)

KFTimes = []
KFMSEs = []
GPTimes = []
GPMSEs= []

trials = 100
for cycle in range(trials):
    kf = makeKF(x,P,F,Q,H,R)
    startKF = time.time()
    xs, cov, _, _ = kf.batch_filter(zs)
    xs, cov, _, _ = kf.rts_smoother(xs, cov)
    KFEnd = time.time() - startKF
    
    # Build the GP kernel from Kalman structure
    startGP = time.time()
    K_clean = gp_kernal(P0, F, Q, H, N)
    K_obs = K_clean + R[0, 0] * np.eye(N)
    mu_post = K_clean @ np.linalg.inv(K_obs) @ zs
    GPEnd = time.time() - startGP
    
    KFTimes.append(KFEnd)
    KFMSEs.append(mean_squared_error(y_pred=xs[:, 0],y_true=track))
    GPTimes.append(GPEnd)
    GPMSEs.append(mean_squared_error(y_pred=mu_post,y_true=track))
    print(f"{cycle+1}/{trials} Cycles done :)")

fig = plt.figure(figsize=(15, 6))

ax1 = fig.add_subplot(121)
ax1.hist(KFTimes, bins=10, alpha=0.6, label='Kalman RTS Times', color='skyblue', edgecolor='black')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel("Frequency")
ax1.set_title("Histrogram of Kalman RTS Times")

ax2 = fig.add_subplot(122)
ax2.hist(GPTimes, bins=10, alpha=0.6, label='Kalman RTS Times', color='salmon', edgecolor='black')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel("Frequency")
ax2.set_title("Histrogram of GP Regression Times")

print(f"KF MSE Average: {sum(KFMSEs)/len(KFMSEs)}")
print(f"GP MSE Average: {sum(GPMSEs)/len(GPMSEs)}")

plt.tight_layout()
plt.show()

fig.savefig("GPAsKalman.png", dpi=300, bbox_inches='tight')