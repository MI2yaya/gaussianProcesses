
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Defined.Helpers.plotting import plotMSE, plotHist 
from Defined.KFs.KF import KalmanFilter

kTrials=50
time=1000
np.random.seed(42)
r=1
q=1
'''
7.1 Scalar random walk
'''
def scalarRandomWalk(trials=10, r=1, q=1):
    x_initial = np.random.normal(0, r)
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

stateErrors = []
measurementErrors = []

for trial in range(kTrials):
    xs, ys = scalarRandomWalk(trials=time,r=r,q=q)

    x = np.array([xs[0]])
    P = np.array([[100]])  # Initial state covariance
    F = np.array([[1.]])  # State transition matrix
    Q = np.array([[1]])  # Process noise covariance
    H = np.array([[1.]])  # Observation matrix
    R = np.array([[1]])  # Measurement noise covariance


    kf = KalmanFilter(x,P,F,H,Q,R)

    Ms, Covs= kf.batch_filter(ys)
    #Ms= kf.rts_smoother(Ms, Covs)

    if trial==0 :
        plotMSE(xs, ys, Ms, r, q, save=False, name="_07p1a_scalar_random_walk.png")

    stateErrors.append(mean_squared_error(xs, Ms))
    measurementErrors.append(mean_squared_error(ys, Ms))

plotHist(stateErrors,measurementErrors, r, q, time, kTrials, save=False, name="_07p1b_scalar_random_walk_errors.png")
raise ValueError
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
xs, ys = constantVelocityModel(trials=time,dt=dt)
measurementErrorsX = []
stateErrorsX = []
measurementErrorsY = []
stateErrorsY = []
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

    stateErrorsX.append(mean_squared_error([x[0] for x in xs], [m[0] for m in Ms]))
    stateErrorsY.append(mean_squared_error([x[2] for x in xs], [m[2] for m in Ms]))
    measurementErrorsX.append(mean_squared_error([y[0] for y in ys], [m[0] for m in Ms]))
    measurementErrorsY.append(mean_squared_error([y[1] for y in ys], [m[2] for m in Ms]))

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
#plt.savefig("_07p2a_constant_velocity_model.png")
#plt.show()

fig = plt.figure(figsize=(12, 6))
plt.title(f"Constant Velocity Model Error; time={time}, trials={kTrials}")
plt.axis("off")
ax1 = fig.add_subplot(221)
ax1.hist(stateErrorsX, bins=30, alpha=0.7)
ax1.set_title("State X")
ax1.set_xlabel("RMSE")
ax1.set_ylabel("Frequency")
ax1.text(0.5, 0.9, f"Mean: {np.mean(stateErrorsX):.3f}\nMedian: {np.median(stateErrorsX):.3f}",
             transform=ax1.transAxes, fontsize=12, verticalalignment='top')
ax2 = fig.add_subplot(222)
ax2.hist(stateErrorsY, bins=30, alpha=0.7)
ax2.set_title("State Y")
ax2.set_xlabel("RMSE")
ax2.set_ylabel("Frequency")
ax2.text(0.5, 0.9, f"Mean: {np.mean(stateErrorsY):.3f}\nMedian: {np.median(stateErrorsY):.3f}",
                transform=ax2.transAxes, fontsize=12, verticalalignment='top')
ax3 = fig.add_subplot(223)
ax3.hist(measurementErrorsX, bins=30, alpha=0.7)
ax3.set_title("Measurement X")
ax3.set_xlabel("RMSE")
ax3.set_ylabel("Frequency")
ax3.text(0.5, 0.9, f"Mean: {np.mean(measurementErrorsX):.3f}\nMedian: {np.median(measurementErrorsX):.3f}", 
         transform=ax3.transAxes, fontsize=12, verticalalignment='top')
ax4 = fig.add_subplot(224)
ax4.hist(measurementErrorsY, bins=30, alpha=0.7)
ax4.set_title("Measurement Y")
ax4.set_xlabel("RMSE")
ax4.set_ylabel("Frequency")
ax4.text(0.5, 0.9, f"Mean: {np.mean(measurementErrorsY):.3f}\nMedian: {np.median(measurementErrorsY):.3f}",
         transform=ax4.transAxes, fontsize=12, verticalalignment='top')
plt.tight_layout()
#plt.savefig("_07p2b_constant_velocity_model_errors.png")
#plt.show()
'''
7.3 Mass-Spring Chain with N identical masses
x = [p1(t) v1(t) ... pn(t) vn(t)]
'''

def massSpringChain(N=3, trials=10, dt=.1, r=1, q=1):
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
xs, ys = massSpringChain(N=N,trials=time)
stateErrors = [] 
measurementErrors = []
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

    for i in range(N):
        stateErrors.append([mean_squared_error([x[i*2] for x in xs], [m[i*2] for m in Ms])])
        measurementErrors.append([mean_squared_error([y[i] for y in ys], [m[i*2] for m in Ms])])

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
#plt.savefig("_07p3a_mass_spring_chain.png")
#plt.show()

fig = plt.figure(figsize=(12, 6))
plt.title(f"Mass-Spring Chain Error; N={N}, time={time}, trials={kTrials}")
plt.axis("off")

for i in range(3):
    ax = fig.add_subplot(231+ i)
    ax.hist(stateErrors[i], bins=30, alpha=0.7)
    ax.set_title(f"State-X #{i+1}")
    ax.set_xlabel("RMSE")
    ax.set_ylabel("Frequency")
    ax.text(0.5, 0.9, f"Mean: {np.mean(stateErrors[i]):.3f}\nMedian: {np.median(stateErrors[i]):.3f}",
             transform=ax.transAxes, fontsize=12, verticalalignment='top')
    ax = fig.add_subplot(231+ i+3)
    ax.hist(measurementErrors[i], bins=30, alpha=0.7)
    ax.set_title(f"Measurement-X #{i+1}")
    ax.set_xlabel("RMSE")
    ax.set_ylabel("Frequency")
    ax.text(0.5, 0.9, f"Mean: {np.mean(measurementErrors[i]):.3f}\nMedian: {np.median(measurementErrors[i]):.3f}",
             transform=ax.transAxes, fontsize=12, verticalalignment='top')
plt.tight_layout()
plt.savefig("_07p3b_mass_spring_chain_errors.png")
plt.show()
