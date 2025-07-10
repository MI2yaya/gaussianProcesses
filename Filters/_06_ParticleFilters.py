from filterpy.monte_carlo import systematic_resample
from numpy.linalg import norm
from numpy.random import uniform, randn
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt

def create_uniform_particles(x_range,y_range,hdg_range,N):
    particles = np.empty((N,3))
    particles[:,0] = uniform(x_range[0],x_range[1],size=N)
    particles[:,1] = uniform(y_range[0],y_range[1],size=N)
    particles[:,2] = uniform(hdg_range[0],hdg_range[1],size=N)
    particles[:,2] %= 2*np.pi
    return particles

def create_gaussian_particles(mean,std,N):
    particles = np.empty((N, 3))
    particles[:, 0] = mean[0] + (randn(N) * std[0])
    particles[:, 1] = mean[1] + (randn(N) * std[1])
    particles[:, 2] = mean[2] + (randn(N) * std[2])
    particles[:, 2] %= 2 * np.pi
    return particles

def predict(particles,u,std,dt=1):
    N=len(particles)
    # update heading
    particles[:, 2] += u[0] + (randn(N) * std[0])
    particles[:, 2] %= 2 * np.pi

    # move in the (noisy) commanded direction
    dist = (u[1] * dt) + (randn(N) * std[1])
    particles[:, 0] += np.cos(particles[:, 2]) * dist
    particles[:, 1] += np.sin(particles[:, 2]) * dist

def update(particles, weights, z, R, landmarks):
    for i, landmark in enumerate(landmarks):
        distance = np.linalg.norm(particles[:, 0:2] - landmark, axis=1)
        weights *= scipy.stats.norm(distance, R).pdf(z[i])

    weights += 1.e-300      # avoid round-off to zero
    weights /= sum(weights) # normalize

def estimate(particles, weights):
    """returns mean and variance of the weighted particles"""

    pos = particles[:, 0:2]
    mean = np.average(pos, weights=weights, axis=0)
    var  = np.average((pos - mean)**2, weights=weights, axis=0)
    return mean, var

def neff(weights):
    return 1. / np.sum(np.square(weights))

def resample_from_index(particles,weights,indexes):
    particles[:] = particles[indexes]
    weights.resize(len(particles))
    weights.fill (1.0 / len(weights))

def run_pf1(N,iters=18,sensor_std_err=.1, initial_x = None,plot=False):
    landmarks =np.array([[-1,2],[5,10],[12,14],[18,21]])
    NL = len(landmarks)

    if initial_x is not None:
        particles = create_gaussian_particles(mean=initial_x,std=(5,5,np.pi/4), N=N)
    else:
        particles = create_uniform_particles((0,20),(0,20),(0,6.28),N)
    weights = np.ones(N)/N

    ys=[]
    xs=[]
    robot_pos = np.array([0.,0.])
    allParticles=[]
    for x in range(iters):
        robot_pos+=(1,1)

        zs = (norm(landmarks-robot_pos,axis=1) + randn(NL)*sensor_std_err)    
        predict(particles,u=(0,1.414),std=(.2,0.05))
        update(particles,weights,z=zs,R=sensor_std_err,landmarks=landmarks)

        if neff(weights)<N/2:
            indexes = systematic_resample(weights)
            resample_from_index(particles,weights,indexes)
            assert np.allclose(weights,1/N)
        mu,var = estimate(particles,weights)
        xs.append(mu)
        ys.append([robot_pos[0],robot_pos[1]])
        allParticles.append([particles[:,0],particles[:,1]])
        if plot:
            plt.scatter(particles[:, 0], particles[:, 1], 
                            color='k', marker=',', s=1)
            p1 = plt.scatter(robot_pos[0], robot_pos[1], marker='+',
                            color='b', s=180, lw=3)
            p2 = plt.scatter(mu[0], mu[1], marker='s', color='r')
    xs = np.array(xs)
    ys = np.array(ys)

    MSE = np.mean(np.square(ys-xs))
    print(f"Pos MSE : {MSE}, Variance: {var}")
    if plot:
        plt.legend([p1, p2], ['Actual', 'PF'], loc=4, numpoints=1)
        plt.show()
    
    return MSE

MSEs=[]
trials=2
N=5000
plot=True
for _ in range(trials):
    MSEs.append(run_pf1(N,initial_x=(1,1,np.pi/4),plot=plot))

print(f"Average MSE: {np.average(MSEs)}")

