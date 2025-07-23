
import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
from sklearn.metrics import mean_squared_error

def plot(xs,ys,Ms,title="",show=True):
    plt.plot(xs, label='True States')
    plt.plot(ys, label='Observations')
    plt.plot(Ms, label='Kalman Filter Estimate')
    plt.title(title)
    plt.xlabel("Time Steps")
    plt.legend()
    plt.show()

kTrials=10

'''
7.1 Scalar random walk
'''
def scalarRandomWalk(trials=10, dt=1,r=1, q=1):
    x_initial = np.random.normal(0, 1)
    xs = [x_initial]
    ys = [x_initial]
    x=x_initial
    for _ in range(trials):
        w = np.random.normal(0, r)
        x= x + w
        xs.append(x)
        y = x + np.random.normal(0, q)
        ys.append(y)
    return xs,ys

errors= []
xs, ys = scalarRandomWalk(trials=kTrials)
for trial in range(kTrials):
    kf = KalmanFilter(dim_x=1, dim_z=1)
    kf.x = np.array([0.])  # initial state
    kf.P = np.eye(1) * 500  # initial covariance
    kf.F = np.array([[1]])  # state transition matrix
    kf.H = np.array([[1]])  # observation matrix
    kf.R = np.array([[1]])  # observation noise covariance
    kf.Q = np.array([[1]])  # process noise covariance

    Xs, Covs, _, _ = kf.batch_filter(ys)
    Ms, Ps, _, _ = kf.rts_smoother(Xs, Covs)

    errors.append(mean_squared_error(xs, Ms))
print(f"Average Scalar Random Walk RMSE over {kTrials} trials: {np.mean(errors):.3f}")

plot(xs, ys, Ms, title="Scalar Random Walk")



'''
7.2 Constant Velocity Model
x = [px,vx,py,vy]
'''
def constantVelocityModel(trials=10, dt=1, r=1, q=1):
    x_initial = np.random.multivariate_normal(np.zeros(4), np.eye(4))
    xs = [x_initial]
    ys = [np.array([x_initial[0], x_initial[2]])]
    x = x_initial
    for _ in range(trials):
        w = np.random.multivariate_normal(np.zeros(4), q * np.eye(4))
        A = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])
        x = A @ x + w
        xs.append(x)
        y = np.random.multivariate_normal(np.zeros(2), r * np.eye(2))
        H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
        y_observed = H @ x + y
        ys.append(y_observed)
    return xs, ys
dt=1
xs, ys = constantVelocityModel(dt=dt)
errors = []
for trial in range(kTrials):
    # Initialize Kalman Filter
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.x = np.zeros(4)  
    kf.P = np.eye(4) * 500 
    kf.F = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])  # state transition matrix
    kf.H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
    kf.R = np.eye(2)
    kf.Q = np.eye(4) * 1 
    Xs, Covs, _, _ = kf.batch_filter(ys)
    Ms, Ps, _, _ = kf.rts_smoother(Xs, Covs)

    errors.append(mean_squared_error(xs, Ms))
print(f"Average Constant Velocity Model RMSE over {kTrials} trials: {np.mean(errors):.3f}")

fig = plt.figure(figsize=(13,10))
plt.axis("off")
plt.title("Constant Velocity Model")
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)
t = np.linspace(0, dt * len(xs), len(xs))
ax1.set_xlabel("Time (s)")
ax2.set_xlabel("Time (s)")
ax3.set_xlabel("Time (s)")
ax4.set_xlabel("Time (s)")

ax1.plot(t, [x[0] for x in xs], label="State", color='blue')
ax1.plot(t, [y[0] for y in ys], label="Obs",color='orange')
ax2.plot(t, [x[2] for x in xs],color='blue')
ax2.plot(t, [y[1] for y in ys],color='orange')
ax1.plot(t, [m[0] for m in Ms],label="Kalman",color='green')
ax2.plot(t, [m[2] for m in Ms],color='green')

ax3.plot(t, [x[1] for x in xs],color='blue')
ax3.plot(t, [m[1] for m in Ms],color='green')
ax4.plot(t, [x[3] for x in xs],color='blue')
ax4.plot(t, [m[3] for m in Ms],color='green')

ax1.set_title("X-Position")
ax2.set_title("Y-Position")
ax3.set_title("X-Velocity")
ax4.set_title("Y-Velocity")
fig.legend(loc="lower right")
plt.show()
'''
7.3 Mass-Spring Chain with N identical masses
x = [p1(t) v1(t) ... pn(t) vn(t)]
'''

def massSpringChain(N=3, trials=10, dt=1, r=1, q=1):
    x_initial = np.random.multivariate_normal(np.zeros(2 * N), np.eye(2 * N))
    xs = [x_initial]
    ys = [np.array([x_initial[i] for i in range(0, 2 * N, 2)])]
    x = x_initial
    for _ in range(trials):
        w = np.random.multivariate_normal(np.zeros(2 * N), q * np.eye(2 * N))
        A = np.eye(2 * N)
        for i in range(N):
            if i > 0:
                A[2*i, 2*i-1] = -dt
            if i < N-1:
                A[2*i+1, 2*i+2] = dt
        x = A @ x + w
        xs.append(x)
        y = np.random.multivariate_normal(np.zeros(N), r * np.eye(N))
        H = np.zeros((N, 2 * N))
        for i in range(N):
            H[i, 2*i] = 1
        y_observed = H @ x + y
        ys.append(y_observed)
    return xs, ys
N = 3
xs, ys = massSpringChain(N=N)
errors = []
for trial in range(kTrials):
    kf = KalmanFilter(dim_x=2 * N, dim_z=N)
    kf.x = np.zeros(2 * N)
    kf.P = np.eye(2 * N) * 500
    kf.F = np.eye(2 * N)
    for i in range(N):
        if i > 0:
            kf.F[2*i, 2*i-1] = -dt
        if i < N-1:
            kf.F[2*i+1, 2*i+2] = dt
    kf.H = np.zeros((N, 2 * N))
    for i in range(N):
        kf.H[i, 2*i] = 1
    kf.R = np.eye(N)
    kf.Q = np.eye(2 * N) * 1
    Xs, Covs, _, _ = kf.batch_filter(ys)
    Ms, Ps, _, _ = kf.rts_smoother(Xs, Covs)

    errors.append(mean_squared_error([x[0] for x in xs], [m[0] for m in Ms]))
print(f"Average Mass Spring Chain RMSE over {kTrials} trials: {np.mean(errors):.3f}")

fig = plt.figure(figsize=(13,8))
plt.axis("off")
plt.title(f"Mass-Spring Chain with {N} Masses")
t= np.linspace(0,dt*len(xs),len(xs))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax1.set_xlabel("Time (s)")
ax2.set_xlabel("Time (s)")
ax3.set_xlabel("Time (s)")
ax1.set_ylabel("Position")
ax2.set_ylabel("Velocity")
ax3.set_ylabel("Position-Observation")
ax1.set_title("Position-State")
ax2.set_title("Velocity-State")
ax3.set_title("Position-Observation")


for s in range(0,N):
    s_xps = [x[2*s] for x in xs]
    s_xvs = [x[2*s+1] for x in xs]
    s_mps = [m[2*s] for m in Ms]
    s_mvs = [m[2*s+1] for m in Ms]
    s_yps = [y[s] for y in ys]
    ax1.plot(t,s_xps, label=f"Mass-Chain #{s+1}")
    ax1.plot(t,s_mps, ls='--', label=f"Kalman Estimate #{s+1}")
    ax2.plot(t,s_xvs)
    ax2.plot(t,s_mvs, ls='--', color='green')
    ax3.plot(t,s_yps)
fig.legend(loc="lower right")
plt.tight_layout()
plt.show()
