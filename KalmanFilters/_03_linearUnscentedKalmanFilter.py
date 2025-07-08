import numpy as np
from numpy.random import randn
from filterpy.kalman import JulierSigmaPoints
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.common import Q_discrete_white_noise
import matplotlib.pyplot as plt

def fx(x, dt): #F matrix, but as a function!
    xout = np.empty_like(x)
    xout[0] = x[1] * dt + x[0]
    xout[1] = x[1]
    return xout

def hx(x): #H matrix, but as a function
    return x[:1] # return position [x] 

sigmas = JulierSigmaPoints(n=2, kappa=1) #montycarlo estimation with sigma points
ukf = UnscentedKalmanFilter(dim_x=2, dim_z=1, dt=1., hx=hx, fx=fx, points=sigmas)
'''
1. Sigma points are passed through
2. Predict with Sigma Points Y=F(x)
    2.a mean calculated through sum(Y(w_im))
    2.b covar calculated through sum(w_ic*(Y-x)(Y-x)^T +Q)
3. Update Z=H(y), mu_z = measurement mean at sigma point
    3.a mean calculated through sum(Z(w_im))
    3.b covar calculated through sum(w_ic*(Z-mu_z)(Z-mu_z)^T +Q)
    3.c Residual: y=z-mu_z
4. Cross Covariance of state and measurements = sum(w_ic*(Y-x)(Z-mu_z)^T)
5. Kalman Gain = (cross Covariance)(posterior Covar)^-1
6. x= x+Ky, P=P-K(posterior Covar)K^T

'''
ukf.P *= 10
ukf.R *= .5
ukf.Q = Q_discrete_white_noise(2, dt=1., var=0.03)

zs, xs = [], []
for i in range(50):
    z = i + randn()*.5
    ukf.predict()
    ukf.update(z)
    xs.append(ukf.x[0])
    zs.append(z)
    
plt.plot(xs)
plt.plot(zs, marker='x', ls='')
plt.show()

#2D Constant Velocity Model now

def f_cv(x, dt):
    F = np.array([[1, dt, 0,  0],
                  [0,  1, 0,  0],
                  [0,  0, 1, dt],
                  [0,  0, 0,  1]])
    return F @ x

def h_cv(x): 
    return x[[0, 2]] 

std_x, std_y = .3, .3
dt = 1.0

sigmas = MerweScaledSigmaPoints(4, alpha=.1, beta=2., kappa=1.)
ukf = UnscentedKalmanFilter(dim_x=4, dim_z=2, fx=f_cv,
          hx=h_cv, dt=dt, points=sigmas)

zs = [np.array([i + randn()*std_x, 
                i + randn()*std_y]) for i in range(100)]      
         
xs, _= ukf.batch_filter(zs)


ukf.x = np.array([0., 0., 0., 0.])
ukf.R = np.diag([0.09, 0.09]) 

ukf.Q[0:2, 0:2] = Q_discrete_white_noise(2, dt=1, var=0.02)
ukf.Q[2:4, 2:4] = Q_discrete_white_noise(2, dt=1, var=0.02)

uxs = []
for z in zs:
    ukf.predict()
    ukf.update(z)
    uxs.append(ukf.x.copy())
uxs = np.array(uxs)


print(f'UKF standard deviation {np.std(uxs - xs):.3f} meters')
plt.plot(xs[:, 0], xs[:, 2])
plt.plot(uxs[:, 0], uxs[:, 2])
plt.show()