from filterpy.kalman import predict, update, KalmanFilter
import math
import numpy as np
from numpy.random import randn
from collections import namedtuple
from scipy.linalg import solve
from filterpy.common import Q_discrete_white_noise
import matplotlib.pyplot as plt
gaussian = namedtuple('Gaussian', ['mean', 'var'])
gaussian.__repr__ = lambda s: f'𝒩(μ={s[0]:.3f}, 𝜎²={s[1]:.3f})'

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


dt=.1 #time scale

#prior
x = np.array([10., 4.5]) #initial mu
P = np.diag([500.,49]) #initial state covar

#state
F = np.array([[1,dt],[0,1]]) #state transition function, Fx = prediction, where pos = pos+dt(vol) and vol = 0(pos)+vol, assuming no vol change
Q = 1 #process variation with mean=0, same to Q in univar

#control
B=0 #control function
u=0 #control input,

#measurement
H = np.array([[1., 0.]]) # Measurement function converts state to measurement, in this case pos (1pos+0vol)
z=1. #measurement noise mean vector
R=np.array([[5.]]) #measurement covar matrix z.dim by z.dim

track, zs = compute_dog_data(R, Q, count=10,dt=dt) #data
xs, cov = [], [] #predictions/cycle
for i,z in enumerate(zs,start=1):
    x , P = predict(x=x,P=P,F=F,Q=Q) #estimate
    x, P = update(x, P, z, R, H) #given measurement, use kalman gain and update!
    xs.append(x)
    cov.append(P)
    print(f"Cycle {i}: Post:{x} as Pos,vol; Covar:{P}, Actual:{track[i-1]}")


#or with KalmanFilter Class :)
def pos_vel_filter(x, P, R, Q=0., dt=1.0):
    """ Returns a KalmanFilter which implements a
    constant velocity model for a state [x dx].T
    """
    
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([x[0], x[1]]) # location and velocity
    kf.F = np.array([[1., dt],
                     [0.,  1.]])  # state transition matrix
    kf.H = np.array([[1., 0]])    # Measurement function
    kf.R *= R                     # measurement uncertainty
    if np.isscalar(P):
        kf.P *= P                 # covariance matrix 
    else:
        kf.P[:] = P               # [:] makes deep copy
    if np.isscalar(Q):
        kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=Q)
    else:
        kf.Q[:] = Q
    return kf

dt = .1
x = np.array([0., 0.]) 
kf = pos_vel_filter(x, P=500, R=5, Q=0.1, dt=dt)

def run(x0=(0.,0.), P=500, R=0, Q=0, dt=1.0, 
        track=None, zs=None,
        count=0, do_plot=True, **kwargs):
    """
    track is the actual position of the dog, zs are the 
    corresponding measurements. 
    """

    # Simulate dog if no data provided. 
    if zs is None:
        track, zs = compute_dog_data(R, Q, count)

    # create the Kalman filter
    kf = pos_vel_filter(x0, R=R, P=P, Q=Q, dt=dt)  

    # run the kalman filter and store the results
    xs, cov = [], []
    for z in zs:
        kf.predict()
        '''
        F=state transition, from prior to measurement
        B=Control function
        u=Control input
        x=Fx+Bu
        
        P=E[(Fx-Fu)(Fx-Fu)^t) = FE[(x-u)(x-u)^t]F^T
        P= FPF^T + Q #prediction uncertainty
        
        '''
        kf.update(z)
        '''
        H = Measurement function converts state to measurement
        R= measurement covar
        S= HPH^T + R  #measurement uncertainty
        
        K= PH^tS^-1 #Kalman Gain, ~ (P/S)H^T ~ uncertaintyPrediction/uncertaintyMeasurement * H^t
        
        x = x+Ky #estimate mean = x +K(z-Hx) = (I-KH)x+Kz #recall x=(1-K)x+kz
        
        P = (I-KH)P #adjusts P based on kalman gain (ratio)
        
        How is velocity updated?
        K_vol = cov(pos,vol)/(var(pos)+var(measurement))
        
        '''
        xs.append(kf.x)
        cov.append(kf.P)

    xs, cov = np.array(xs), np.array(cov)

    return xs, cov

P = np.diag([500., 49.]) #state covar
Ms, Ps = run(count=50, R=10, Q=0.01, P=P)


P = np.diag([500., 49.])
f = pos_vel_filter(x=(0., 0.), R=3., Q=.02, P=P)
track, zs = compute_dog_data(3., .02, count=50)
Xs, Covs, _, _ = f.batch_filter(zs) #does all measurements at once, is really just a loop lol
Ms, Ps, _, _ = f.rts_smoother(Xs, Covs) #smooths means, covariances, and kalman gains 

#compare kalman velocity vs rts velocity
dx = np.diff(Xs[:, 0], axis=0)
plt.scatter(range(1, len(dx) + 1), dx, facecolor='none', 
            edgecolor='k', lw=2, label='Raw velocity') 
plt.plot(Xs[:, 1], ls='--', label='Kalman Velocity')
plt.plot(Ms[:, 1], label='RTS Velocity')
plt.legend(loc=4)
plt.gca().axhline(1, lw=1, c='k')
plt.show()

