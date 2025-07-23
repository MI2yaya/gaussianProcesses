import math
import numpy as np
import matplotlib.pyplot as plt

def plot(xs,dt,ys=None,title="Test",show=True):
    t= np.linspace(0,dt*len(xs),len(xs))

    plt.plot(t,xs, label="states")
    if ys:
        plt.plot(t,ys, label="observations")
    if show:
        plt.legend()
        plt.show()

'''
7.1 Scalar random walk
'''
dt=1
trials=99

def nextState(x,q=1):
    w = np.random.normal(0,q)
    return x+w

def observation(x,r=1):
    v = np.random.normal(0,r)
    return(x+v)
x = np.random.normal(0,1)

xs=[0]
ys=[0]
for _ in range(trials):
    x = nextState(x)
    xs.append(x)
    y = observation(x)
    ys.append(y)
#plot(xs,dt,ys,title="Scalar Random Walk")

'''
7.2 Constant Velocity Model
x = [px,vx,py,vy]
'''
def nextState(x,dt,q=1):
    w=q*np.eye(4)
    A = np.array([[1,dt,0,0],[0,1,0,0],[0,0,1,dt],[0,0,0,1]]) #how state transitions
    x2 = A @ x
    return x2 + w

def observation(x,r=1):
    v = r*np.random.randn(2,1)
    H = np.array([[1,0,0,0],[0,0,1,0]]) #how state => observation (px,py)
    y = H @ x
    return y+v

dt=1
trials=10
x = np.eye(4)
xs=[x]
ys=[x]
for _ in range(trials):
    x = nextState(x,dt)
    xs.append(x)

    y = observation(x)
    ys.append(y)

print([x[0][0] for x in xs])
#plot([x[0][0] for x in xs],dt,[y[0][0] for y in ys],title="Constant-Velocity Model x-Pos")

'''
7.3 Mass-Spring Chain with N identical masses
x = [p1(t) v1(t) ... pn(t) vn(t)]
'''

def nextState(x,dt,N,q=1):
    w = np.random.multivariate_normal(np.zeros(2*N),q*np.eye(2*N))
    K = 2*np.eye(N) -1*np.eye(N,k=1) -1*np.eye(N,k=-1)
    A = np.block([[np.eye(N), dt*np.eye(N)],[-dt*K,np.eye(N)]])
    x = A @ x
    return x+w

def observation(y,N,r=1):
    v = np.random.multivariate_normal(np.zeros(N),r*np.eye(N))
    H = np.block([[np.eye(N), np.zeros((N,N))]])
    y = H @ y
    return y + v

N=10
dt=1
trials=10
x = np.random.multivariate_normal(np.zeros(2*N),np.eye(2*N))
xs=[x]
ys=[x]
for _ in range(trials):
    x = nextState(x,dt,N)
    xs.append(x)

    y = observation(x,N)
    ys.append(y)

fig = plt.figure(figsize=(13,8))

plt.title(f"Mass-Spring Chain with {N} Masses")
t= np.linspace(0,dt*len(xs),len(xs))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax1.set_title("Position-State")
ax2.set_title("Velocity-State")
ax3.set_title("Position-Observation")


for s in range(0,N):
    s_xps = [x[2*s] for x in xs]
    s_xvs = [x[2*s+1] for x in xs]
    s_yps = [y[s] for y in ys]
    ax1.plot(t,s_xps, label=f"#{s}")
    ax2.plot(t,s_xvs)
    ax3.plot(t,s_yps)
fig.legend()
plt.tight_layout()
plt.show()