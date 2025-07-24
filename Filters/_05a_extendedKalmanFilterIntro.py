import numpy as np
from numpy import eye, array, asarray
from math import sqrt
from numpy.random import randn
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import ExtendedKalmanFilter
import matplotlib.pyplot as plt
'''
1. System
    Airplane moves straight line, no change in y, and gives radius and bearing
    x = [x,vx,y]
    F is straight forward
    H converts from X to radius, bearing
'''

class Airplane:
    def __init__(self, dt, pos, vel, alt):
        self.pos = pos
        self.vel = vel
        self.alt = alt
        self.dt = dt
        self.track=[]
        self.readings=[]
        
    def get_range(self):
        """ Returns slant range to the object. Call once 
        for each new measurement at dt time from last call.
        """
        
        # add some process noise to the system
        velReading = self.vel  + .1*randn()
        altReading = self.alt + .1*randn()
        posReading = (self.pos + .1*randn())+ velReading*self.dt
        
        self.pos = self.pos + self.vel*self.dt
        
        self.track.append((self.pos,self.vel,self.alt))
        self.readings.append((posReading,velReading,altReading))
    
        # add measurement noise
        err = self.pos * 0.05*randn()
        slant_dist = sqrt(self.pos**2 + self.alt**2)
        
        
        return slant_dist + err


def f(x,dt):
    '''
    We can find this mathematically with [xv,xa] = A[x,xv], where A = [[0,1],[0,0]]
    Then, compute F through power series expansion. F(dt) = e^(A*dt) = I + A*dt + (A*dt)^2/2! ...
    A^2 = [[0,0],[0,0]], so F(dt) = I +A*dt + 0 
    so F = [[1,0],[0,1]] + [[0,1],[0,0]]dt = [[1,dt],[0,1]]
    '''
    F = np.array([[1,dt,0],
                [0,1,0],
                [0,0,1]])
    return F @ x

def HJacobian_at(x):
    '''
    We know h(x) = (x^2+y^2)^1/2
    Therefore, H = ∂h(x)/∂x |x_t
    H = [∂h/∂x, ∂h/∂xv, ∂h/y]
    ∂h/∂x = x/(x^2+y^2)^1/2
    ∂h/∂xv = 0
    ∂h/∂y = y/(x^2+y^2)^1/2
    '''
    horiz_dist = x[0]
    altitude   = x[2]
    denom = sqrt(horiz_dist**2 + altitude**2)
    return  array ([[horiz_dist/denom, 0., altitude/denom]])

def h(x):
    return (x[0]**2 + x[2]**2) ** 0.5

dt = 0.05
rk = ExtendedKalmanFilter(dim_x=3, dim_z=1)
radar = Airplane(dt=dt, pos=0., vel=100., alt=1000.)

rk.x = array([radar.pos-100, radar.vel+100, radar.alt+1000])

rk.F = eye(3) + array([[0, 1, 0],
                       [0, 0, 0],
                       [0, 0, 0]]) * dt

range_std = 5. # meters
rk.R = np.diag([range_std**2])
rk.Q[0:2, 0:2] = Q_discrete_white_noise(2, dt=dt, var=0.1)
rk.Q[2,2] = 0.1
rk.P *= 50

xs = []
for i in range(int(20/dt)):
    z = radar.get_range()
    
    rk.update(array([z]), HJacobian_at, h)
    xs.append(rk.x)
    rk.predict()

track = radar.track
readings = radar.readings

xs = asarray(xs)
track = asarray(track)
readings = asarray(readings)

time = np.arange(0, len(xs)*dt, dt)
print(f'UKF standard deviation {np.std(track[:,[0,2]]- xs[:,[0,2]]):.3f} meters')

plt.figure(figsize=(12, 5))

# plot
plt.subplot(1, 3, 1)
plt.plot(time, xs[:, 0], label='UKF Estimated X')
plt.plot(time,track[:,0], '--', label='Actual X')
plt.plot(time,readings[:,0], label='Actual+Error X')
plt.xlabel('Time')
plt.ylabel('x pos')
plt.legend()
plt.title('X Plane')

plt.subplot(1, 3, 2)
plt.plot(time, xs[:, 1], label='UKF Estimated XV')
plt.plot(time,track[:,1], '--', label='Actual XV')
plt.plot(time,readings[:,1], label='Actual+Error XV')
plt.xlabel('Time')
plt.ylabel('x pos')
plt.legend()
plt.title('XV Plane')

plt.subplot(1, 3, 3)
plt.plot(time, xs[:, 2], label='UKF Estimated Y')
plt.plot(time,track[:,2], '--', label='Actual Y')
plt.plot(time,readings[:,2], label='Actual+Error Y')
plt.xlabel('Time')
plt.ylabel('x pos')
plt.legend()
plt.title('Y Plane')


plt.tight_layout()
plt.show()