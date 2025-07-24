import numpy as np
from numpy import eye, array, asarray
from math import sqrt, atan2
from numpy.random import randn
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import ExtendedKalmanFilter
import matplotlib.pyplot as plt

'''
System
    Robot moves with velocity and steering angle
    X = [x,y,theta]
    U = [xv,alpha(steering angle)]
'''

'''
F:
Same strategy as before, we know X = f(x,u) + N(0,Q), f(x,u) = 
        [[x-r*sympy.sin(theta) + r*sympy.sin(theta+beta)],
        [y+r*sympy.cos(theta)- r*sympy.cos(theta+beta)],
        [theta+beta]])
        
The Jacobian of f(x,u) against x = 
        [1,0,0] +[0,0,-R*cos(theta)+Rcos(theta+beta)],
        [0,1,0] +[0,0,-R*sin(theta)+R*sin(theta+beta)],
        [0,0,1] +[0,0,0]

M: [var_vel,0],[0,var_alpha]        

V:
        Jacobian of f(x,u) againt u = 
        a very long matrix I wont type

P = FPF^T + VMV^T
'''

'''
Measurement
        Noisy bearing and range measurements
        r = sqrt((p_x-x)^2+(p_y-y)^2) given p position of landmark
        theta = arctan((p_y-y)/(p_x-x))-theta
        so Z = h(x,p)=[r,theta] + N(0,R)
        linearize Z with jacobian at x, also a long matrix defined next
'''


def h_of(x,pos):
    hyp = (pos[0]-x[0,0])**2+(pos[1]-x[1,0])**2
    dist = sqrt(hyp)
    H = array(
        [[-(pos[0] - x[0, 0]) / dist, -(pos[1] - x[1, 0]) / dist, 0],
         [ (pos[1] - x[1, 0]) / hyp,  -(pos[0] - x[0, 0]) / hyp, -1]])
    return H

def Hx(x,pos):
    hyp = (pos[0]-x[0,0])**2+(pos[1]-x[1,0])**2
    dist = sqrt(hyp)
    Hx = array([[dist]],
               [atan2(pos[1]-x[1,0],pos[0]-x[0,0]-x[2,0])])