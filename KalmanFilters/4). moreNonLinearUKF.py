import numpy as np
from numpy.random import randn
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
'''
1. Define a system 
    Car moves and gives reading as radius and bearing
    x = [x,vx,y,vy]
    F is straight forward
    H is from state to measurement



'''
class Car:
    def __init__(self, init_state, std_range, std_bearing, dt=1.0):
        self.state = np.array(init_state, dtype=float)  # [x, vx, y, vy]
        self.dt = dt
        self.std_range=std_range
        self.std_bearing=std_bearing
        self.trueList=[]
        self.F = np.array([[1, dt, 0,  0],
                           [0,  1, 0,  0],
                           [0,  0, 1, dt],
                           [0,  0, 0,  1]])

    def move(self):
        """Updates the true state using constant velocity model"""
        self.state = self.F @ self.state
        

    def sense(self):
        """Returns noisy [range, bearing] radar measurement"""
        x, vx, y, vy = self.state
        r = np.sqrt(x**2 + y**2)
        b = np.arctan2(y, x)
        self.trueList.append([x, y])
        
        r += randn() * self.std_range
        b += randn() * self.std_bearing
        return np.array([r, b])
    
    def history(self):
        return np.array(self.trueList)
       
def residual_z(a, b):
    """Residual for measurement: a - b, handling angle wraparound"""
    y = a - b
    y[1] = np.arctan2(np.sin(y[1]), np.cos(y[1]))  # wrap angle to [-pi, pi]
    return y

def mean_z(sigmas, Wm):
    """Mean of sigma points in measurement space"""
    mean = np.zeros(2)
    mean[0] = np.dot(Wm, sigmas[:, 0])  # range is linear

    # bearing is circular — take weighted average using sine & cosine
    sin_sum = np.dot(Wm, np.sin(sigmas[:, 1]))
    cos_sum = np.dot(Wm, np.cos(sigmas[:, 1]))
    mean[1] = np.arctan2(sin_sum, cos_sum)
    return mean     

#x = [x_p, x_v, y_p, y_v]

# State transition function
def fx(x,dt):
    F = np.array([[1,dt,0,0],
                [0,1,0,0],
                [0,0,1,dt],
                [0,0,0,1]])
    return F @ x

#measurement function
def hx(x):
    px,vs,py,vy = x
    r = np.sqrt(px**2 + py**2)
    b = np.arctan2(py,px)
    return np.array([r, b])
   
dt=1   
n=4
points = MerweScaledSigmaPoints(n,alpha=1.,beta=2,kappa=1)
ukf = UnscentedKalmanFilter(4,2,dt,hx,fx,points,z_mean_fn=mean_z,residual_z=residual_z)

# Initial
ukf.x = np.array([0,0,0,0])
ukf.P*=1

#state
ukf.Q[0:2, 0:2] = Q_discrete_white_noise(dim=2, dt=dt, var=0.1) 
ukf.Q[2:4, 2:4] = Q_discrete_white_noise(dim=2, dt=dt, var=0.1)

#measurement
ukf.R = np.diag([0.3**2, 0.1**2])


car = Car([0, 1.0, 0, 1.0],.3,.1,dt)

zs=[]
for _ in range(100):
    car.move()
    zs.append(car.sense())
    
zs = np.array(zs)

xs, ps= ukf.batch_filter(zs)
xs, ps, K = ukf.rts_smoother(xs, ps)


print(f'UKF standard deviation {np.std(car.history()- xs[:,[0,2]]):.3f} meters')
plt.figure(figsize=(12, 5))

# Cartesian
plt.subplot(1, 2, 1)
plt.plot(xs[:, 0], xs[:, 2], label='UKF Estimated Path')
plt.plot(*np.array(car.history()).T, '--', label='True Path')
plt.xlabel('x position')
plt.ylabel('y position')
plt.legend()
plt.title('XY Plane')

# Polar radar space
plt.subplot(1, 2, 2, projection='polar')
plt.scatter(zs[:, 1], zs[:, 0], alpha=0.3, label='Radar Measurement')
plt.title("Radar Measurements (Range/Bearing)")
plt.legend()

plt.tight_layout()
plt.show()