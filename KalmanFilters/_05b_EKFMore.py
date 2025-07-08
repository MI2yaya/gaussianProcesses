import numpy as np
from numpy import eye, array, asarray
from math import sqrt
from numpy.random import randn
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import ExtendedKalmanFilter
import matplotlib.pyplot as plt

import sympy
from sympy.abc import alpha, x, y, v, w, R, theta
from sympy import symbols, Matrix
'''
1. System
    Robot moves with velocity and steering angle
    X = [x,y,theta]
    U = [xv,alpha(steering angle)]
'''

sympy.init_printing(use_latex="mathjax", fontsize='16pt')
time = symbols('t')
d = v*time
beta = (d/w)*sympy.tan(alpha)
r = w/sympy.tan(alpha)

fxu = Matrix([[x-r*sympy.sin(theta) + r*sympy.sin(theta+beta)],
              [y+r*sympy.cos(theta)- r*sympy.cos(theta+beta)],
              [theta+beta]])
F = fxu.jacobian(Matrix([x, y, theta]))
B, R = symbols('beta, R')
F = F.subs((d/w)*sympy.tan(alpha), B)
F.subs(w/sympy.tan(alpha), R)
print(F)

V = fxu.jacobian(Matrix([v, alpha]))
V = V.subs(sympy.tan(alpha)/w, 1/R) 
V = V.subs(time*v/R, B)
V = V.subs(time*v, 'd')
print(V)


def f(x,dt):
    '''
    Same strategy as before, we know X = f(x,u) + N(0,Q), f(x,u) = 
            [[x-r*sympy.sin(theta) + r*sympy.sin(theta+beta)],
            [y+r*sympy.cos(theta)- r*sympy.cos(theta+beta)],
            [theta+beta]])
              
    The Jacobian of f(x,u) = 
            [[1,0,-R*cos(theta)+Rcos(theta+beta)],
            [0,1,-R*sin(theta)+R*sin(theta+beta)],
            [0,0,1]]
    '''
    F = np.array([[1,dt,0],
                [0,1,0],
                [0,0,1]])
    return F @ x
