import numpy as np
import random
import matplotlib.pyplot as plt
from collections import namedtuple

gaussian = namedtuple('Gaussian', ['mean', 'var'])
gaussian.__repr__ = lambda s: f'𝒩(μ={s[0]:.3f}, 𝜎²={s[1]:.3f})'

def volt(voltage, var):
    return voltage + (random.random() * (var**.5))

def update(prior,measurement):
    x,P = prior
    z, R = measurement
    
    y = z-x #residual
    K = P/(P+R) #calculate kalman gain
    
    x = x + K*y #posterior
    P= (1-K)*P #Posterior var
    return gaussian(x,P) #posterior

def predict(posterior, movement):
    x, P = posterior # mean and variance of posterior
    dx, Q = movement # mean and variance of movement
    x = x + dx
    P = P + Q
    return gaussian(x, P)

temp_change = 1 #temp changes by 1
temp_var= 30000 #30000 degree var
starting_temp = 16.3

N = 1000
zs = [volt(starting_temp, temp_var)+temp_change*(i+1) for i in range(N)]
estimates = []

process_var = 2 #var in measurement
process_model = gaussian(0., process_var)
x = gaussian(zs[0], 500.) # initial state, set initial to the first measurement, pretty smart!

for z in zs:
    prior = predict(x, process_model) #estimate
    x = update(prior, gaussian(z, temp_var)) #based on prior and measurement w/ measurement uncertainty, find posterior

    # save for latter plotting
    estimates.append(x.mean)


plt.plot(estimates)
plt.plot(zs)
plt.title('means w/ raw code')
plt.show()
print(f'means converges to {estimates[-1]:.3f}')


#with filterpy
import filterpy.kalman as kf

x= zs[0] #initial mean
P = 500 #initial var
u = 0 #process mean
Q = 2 #process var

R=30000 #measurement var
estimates = []
for z in zs:
    x,P = kf.predict(x=x, P=P, u=u, Q=Q) #Estimate given N(x,P)+N(u,Q) = N(x+u,P+Q)
    x,P = kf.update(x=x,P=P,z=z,R=R) #Update estimate with Kalman Gain given N(x,P) and N(z,R) = N(x+K(z-x),(1-K)P) where K is a normalizing ratio valuing small std

    estimates.append(x)
    
plt.plot(estimates)
plt.plot(zs)
plt.title('means w/ filterPy')
plt.show()
print(f'means converges to {estimates[-1]:.3f}')