import numpy as np
import random
import matplotlib.pyplot as plt
from collections import namedtuple
from numpy.random import randn
import math

def compute_dog_data(z_var, process_var, count=1, dt=1.):
    "returns track, measurements 1D ndarrays"
    print(z_var)
    print(process_var)
    x, vel = 0., 1.
    z_std = math.sqrt(z_var) 
    p_std = math.sqrt(process_var)
    xs, zs = [], []
    for _ in range(count):
        v = vel + (randn() * p_std)
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

def kalman_predict(x,P,F,Q):
    x = F @ x
    P = F @ P @ F.T + Q
    return(x,P)


def update(x,P,z,R,H):
    y = z - H @ x
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    x = x +K @ y
    P = (np.eye(len(x)) - K@H)@P
    return x,P
    
dt=.1 #time scale
N=100
#initial belief
x = np.array([0, 0])    #state: [pos,vel]
P = np.diag([500.,49])      #state covar
P0 = np.diag([500.,49]) 

#state
A = np.array([[1,dt],[0,1]]) #state transition matrix
Q = np.array([[0.01, 0.],
              [0., 0.01]])   #process noise covar

#measurement
H = np.array([[1., 0.]])     #state to measurement
R=np.array([[1]])           #measurement noise covar

track, zs = compute_dog_data(R[0, 0], Q[0, 0], count=N,dt=dt) #data
xs, cov = [], [] #predictions/cycle
for i,z in enumerate(zs,start=1):
    x, P = kalman_predict(x=x,P=P,F=A,Q=Q) #estimate
    x, P = update(x, P, z, R, H) #given measurement, use kalman gain and update!
    xs.append(x)
    cov.append(P)
    print(f"Cycle {i}: Post:{x} as Pos,vol; Covar:{P}, Actual:{track[i-1]}")


# Build the GP kernel from Kalman structure
K_clean = gp_kernal(P0, A, Q, H, N)
K_obs = K_clean + R[0, 0] * np.eye(N)
mu_post = K_clean @ np.linalg.inv(K_obs) @ zs

xs = np.array(xs)
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