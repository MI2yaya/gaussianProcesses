
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Defined.Helpers.plotting import plotMSE, plotHist 
from Defined.KFs.KF import KalmanFilter

kTrials=2
time=1000
#np.random.seed(42)
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
xs_list=[]
ys_list=[]
MsX_list=[]
MsY_list=[]
for trial in range(kTrials):
    xs, ys = scalarRandomWalk(trials=time,r=r,q=q)
    x = np.array([xs[0]])
    P = np.array([[100]])  # Initial state covariance
    F = np.array([[1.]])  # State transition matrix
    Q = np.array([[q]])  # Process noise covariance
    H = np.array([[1.]])  # Observation matrix
    R = np.array([[r]])  # Measurement noise covariance


    kf = KalmanFilter(x,P,F,H,Q,R)

    Ms, Covs= kf.batch_filter(ys)

    MsX = [ele + np.random.normal(0, q) for ele in Ms]
    MsY = [ele + np.random.normal(0, r) for ele in Ms]

    stateErrors.append(mean_squared_error(xs, MsX))
    measurementErrors.append(mean_squared_error(ys, MsY))
    xs_list.append(xs)
    ys_list.append(ys)
    MsX_list.append(MsX)
    MsY_list.append(MsY)


plotMSE(np.median(xs_list,axis=0), np.median(ys_list,axis=0), np.median(MsX_list,axis=0),np.median(MsY_list,axis=0), r, q, save=False, name="_07p1a_median_scalar_random_walk.png")
plotHist(stateErrors,measurementErrors, r, q, time, kTrials, save=False, name="_07p1b_scalar_random_walk_errors.png")

'''
7.2 Constant Velocity Model
x = [px,vx,py,vy] #generalize into N targets
'''
kTrials=10
time=100
r=10
q=10
dt=1

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

measurementErrorsX = []
stateErrorsX = []
measurementErrorsY = []
stateErrorsY = []
xs_list = []
ys_list = []
for trial in range(kTrials):
    xs, ys = constantVelocityModel(trials=time,dt=dt,r=r,q=q)
    x = np.zeros(4)  
    P = np.eye(4) * 100
    F = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])
    H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
    R = np.eye(2) * r
    Q = np.eye(4) * q 
    kf = KalmanFilter(x, P, F, H, Q, R)

    Ms, Covs = kf.batch_filter(ys)

    MsX = [ele + np.random.normal(0, q) for ele in Ms]
    MsY = [ele + np.random.normal(0, r) for ele in Ms]

    stateErrorsX.append(mean_squared_error([x[0] for x in xs], [m[0] for m in MsX]))
    stateErrorsY.append(mean_squared_error([x[2] for x in xs], [m[2] for m in MsY]))
    measurementErrorsX.append(mean_squared_error([y[0] for y in ys], [m[0] for m in MsX]))
    measurementErrorsY.append(mean_squared_error([y[1] for y in ys], [m[2] for m in MsY]))
    xs_list.append(xs)
    ys_list.append(ys)


plotMSE(np.median(xs_list,axis=0), np.median(ys_list,axis=0), [m[0] for m in MsX], [m[0] for m in MsY], r=r, q=q, save=False, name="_07p2a_constant_velocity_model.png",title="Constant Velocity Model X")
plotMSE(np.median(xs_list,axis=2), np.median(ys_list,axis=1), [m[2] for m in MsX], [m[2] for m in MsY], r=r, q=q, save=False, name="_07p2b_constant_velocity_model_y.png",title="Constant Velocity Model Y")
plotHist(stateErrorsX, measurementErrorsX, r, q, time, kTrials, save=False, name="_07p2c_constant_velocity_model_x_errors.png",title="Constant Velocity Model X Errors")
plotHist(stateErrorsY, measurementErrorsY, r, q, time, kTrials, save=False, name="_07p2d_constant_velocity_model_y_errors.png",title="Constant Velocity Model Y Errors")


'''
7.3 Mass-Spring Chain with N identical masses
x = [p1(t) v1(t) ... pn(t) vn(t)]
'''
kTrials=10
time=100
r=10
q=10
dt=.1
N=3

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
stateErrors = [] 
measurementErrors = []
for trial in range(kTrials):
    xs, ys = massSpringChain(N=N,trials=time,q=q,r=r,dt=dt)
    x = np.zeros(2 * N)
    P = np.eye(2 * N) * 100
    F = np.eye(2 * N)
    for i in range(N):
        if i > 0:
            F[2*i, 2*i-1] = -dt
        if i < N-1:
            F[2*i+1, 2*i+2] = dt
    H = np.zeros((N, 2 * N))
    for i in range(N):
        H[i, 2*i] = 1
    R = np.eye(N) * r
    Q = np.eye(2 * N) * q

    kf = KalmanFilter(x, P, F, H, Q, R)


    Ms, Covs= kf.batch_filter(ys)
    MsX = [ele + np.random.normal(0, q) for ele in Ms]
    MsY = [ele + np.random.normal(0, r) for ele in Ms]



    for i in range(N):
        stateErrors.append([mean_squared_error([x[i*2] for x in xs], [m[i*2] for m in MsX])])
        measurementErrors.append([mean_squared_error([y[i] for y in ys], [m[i*2] for m in MsY])])
    if trial==0:
        for i in range(N):
            fig = plt.figure(figsize=(10, 6))
            ax1 = fig.add_subplot(31+N*100+1*(i+1))
            ax2 = fig.add_subplot(31+N*100+2*(i+1))
            ax3 = fig.add_subplot(31+N*100+3*(i+1))
            ax1.set_title(f"{i+1}th Mass-Spring Chain")
            ax1.plot(xs[2*i], label='True States', color='blue')
            ax1.plot(ys[i], label='Observations', color='orange')
            ax1.plot(MsY[2*i], label='KF Estimate (Observation)', color='green')
            ax1.set_title("State vs Observation vs Estimate")
            ax1.set_xlabel("Time Steps")
            ax1.legend()
            ax2.set_title("State MSE over Time")

            ax2.plot([mean_squared_error(xs[:j+1][2*i], MsX[:j+1][2*i]) for j in range(len(Ms))], label='MSE', color='blue')
            ax2.plot([r]*len(xs), label=f'r={r:.1f}', color='red', linestyle='--')
            ax2.set_xlabel("Time Steps")
            ax2.set_ylabel("MSE")
            ax2.legend()
            ax3.set_title("Measurement MSE over Time")

            for K in range(10):
                print(f"YS:{[y[i] for y in ys[:K]]}")
                print(f"MsY:{[my[i] for my in MsY[:K]]}")


            ax3.plot([mean_squared_error(ys[:j+1][i], MsY[:j+1][2*i]) for j in range(len(Ms))], label='MSE', color='blue')
            ax3.plot([q]*len(ys), label=f'q={q:.1f}', color='red', linestyle='--')
            ax3.set_xlabel("Time Steps")
            ax3.set_ylabel("MSE")
            ax3.legend()

        plt.tight_layout()
        plt.show()


for i in range(N):
    plotHist([se[i][0] for se in stateErrors], [me[i][0] for me in measurementErrors], r, q, time, kTrials, save=False, name=f"_07p3b_mass_spring_chain_{i}_errors.png", title=f"Mass-Spring Chain {i} Errors")
