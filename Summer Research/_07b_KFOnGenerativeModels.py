
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Defined.Helpers.plotting import plotMSE, plotHist 
from Defined.KFs.KF import KalmanFilter
from PIL import Image


kTrials=0
time=1000
np.random.seed(42)
save=False
r=5
q=5
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
    x = np.array([ys[0]])
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

if kTrials>0:
    plotMSE(xs_list, ys_list, MsX_list, MsY_list, r, q, save=save, title=f"S.R.W, Trials:{kTrials},time:{time},r:{r},q{q}",name="_07p1a_median_scalar_random_walk.png")
    plotHist(stateErrors,measurementErrors, r, q, time, kTrials, save=save, title=f"S.R.W, Trials:{kTrials},time:{time},r:{r},q{q}",name="_07p1b_scalar_random_walk_errors.png")

'''
7.2 Constant Velocity Model
x = [px,vx,py,vy] #generalize into N targets
'''
np.random.seed(42)
kTrials=0
time=100
r=5
q=5
dt=1
trackers=2
save=False

def constantVelocityModel(trials=10, dt=1, r=1, q=1,trackers=1):
    x_initial = np.random.multivariate_normal(np.zeros(4*trackers), np.eye(4*trackers))
    xs = [x_initial]
    ys = [np.array([x_initial[i] for i in range(0, 4*trackers, 2)])]
    x = x_initial
    for _ in range(trials):
        w = np.random.multivariate_normal(np.zeros(4*trackers), q * np.eye(4*trackers))
        A = np.eye(4*trackers)
        for i in range(0,trackers*4,2):
            A[i][i+1]=dt
        
        #print(f'X:{x}\nA:{A}\nw:{w}')
        
        x = A @ x + w
        xs.append(x)
        y = np.random.multivariate_normal(np.zeros(2*trackers), r * np.eye(2*trackers))
        
        H= np.zeros((2*trackers, 4*trackers))
        for i in range(0,trackers*2,1):
            H[i][2*i]=1
        #print(f'Y:{y}\nH:{H}')
        
        y_observed = H @ x + y
        ys.append(y_observed)
    return xs, ys

measurementErrorsX = []
stateErrorsX = []
measurementErrorsY = []
stateErrorsY = []
xs_list = []
ys_list = []
MsX_list = []
MsY_list = []
for trial in range(kTrials):
    xs, ys = constantVelocityModel(trials=time,dt=dt,r=r,q=q,trackers=trackers)
    x = np.zeros(4*trackers)  
    P = np.eye(4*trackers) * 100
    
    F = np.eye(4*trackers)
    for i in range(0, trackers*4, 2):
        F[i][i+1] = dt
    
    H = np.zeros((2*trackers, 4*trackers))
    for i in range(0, trackers*2, 1):
        H[i][2*i] = 1
    
    R = np.eye(2*trackers) * r
    Q = np.eye(4*trackers) * q 
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
    MsX_list.append(MsX)
    MsY_list.append(MsY)

#xs_list [trial][time_step][state]
if kTrials>0:
    for tracker in range(trackers):
        #X-pos plotting for tracker
        trackerX = [[x[0*tracker] for x in xs] for xs in xs_list]
        trackerY = [[y[0*tracker] for y in ys] for ys in ys_list]
        trackerMsX = [[m[0*tracker] for m in Ms] for Ms in MsX_list]
        trackerMsY = [[m[0*tracker] for m in Ms] for Ms in MsY_list]
        plotMSE(trackerX, trackerY, trackerMsX, trackerMsY, r=r, q=q, save=save, name=f"_07p2a_tracker{tracker}_constant_velocity_model.png",title=f"C.V.M-X Tracker #{tracker+1}/{trackers} Trials:{kTrials},time:{time},r:{r},q{q}")
        
        #Y-pos plotting for tracker
        trackerX = [[x[2*tracker] for x in xs] for xs in xs_list]
        trackerY = [[y[1*tracker] for y in ys] for ys in ys_list]
        trackerMsX = [[m[2*tracker] for m in Ms] for Ms in MsX_list]
        trackerMsY = [[m[2*tracker] for m in Ms] for Ms in MsY_list]
        plotMSE(trackerX, trackerY,trackerMsX, trackerMsY, r=r, q=q, save=save, name=f"_07p2b_tracker{tracker}_constant_velocity_model.png",title=f"C.V.M-Y Tracker #{tracker+1}/{trackers} Trials:{kTrials},time:{time},r:{r},q{q}")
        
        
        plotHist(stateErrorsX, measurementErrorsX, r, q, time, kTrials, save=save, name=f"_07p2c_tracker{tracker}_constant_velocity_model.png",title=f"C.V.M X Errors Tracker #{tracker+1}/{trackers}")
        plotHist(stateErrorsY, measurementErrorsY, r, q, time, kTrials, save=save, name=f"_07p2d_tracker{tracker}_constant_velocity_model.png",title=f"C.V.M Y Errors Tracker #{tracker+1}/{trackers}")
        if save:
            img1_path = f"_07p2a_tracker{tracker}_constant_velocity_model.png"
            img2_path = f"_07p2b_tracker{tracker}_constant_velocity_model.png"
            img3_path = f"_07p2c_tracker{tracker}_constant_velocity_model.png"
            img4_path = f"_07p2d_tracker{tracker}_constant_velocity_model.png"
            img1 = Image.open(img1_path)
            img2 = Image.open(img2_path)
            img3 = Image.open(img3_path)
            img4 = Image.open(img4_path)
            combined = Image.new('RGB', (img1.width + img2.width, img1.height + img3.height))
            combined.paste(img1, (0, 0))
            combined.paste(img2, (img1.width, 0))
            combined.paste(img3, (0, img1.height))
            combined.paste(img4, (img1.width, img1.height))
            combined.save(f"_07p2_tracker{tracker}_constant_velocity_model_combined.png")
            img1.close()
            img2.close()
            img3.close()
            img4.close()
            os.remove(img1_path)
            os.remove(img2_path)
            os.remove(img3_path)
            os.remove(img4_path)
        


'''
7.3 Mass-Spring Chain with N identical masses
x = [p1(t) v1(t) ... pn(t) vn(t)]
'''
kTrials=0
time=100
r=10
q=10
dt=.1
N=3
save=False

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
xs_list=[]
ys_list=[]
MsX_list=[]
MsY_list=[]

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

    xs_list.append(xs)
    ys_list.append(ys)
    MsX_list.append(MsX)
    MsY_list.append(MsY)


    for i in range(N):
        stateErrors.append([mean_squared_error([x[i*2] for x in xs], [m[i*2] for m in MsX])])
        measurementErrors.append([mean_squared_error([y[i] for y in ys], [m[i*2] for m in MsY])])
if kTrials>0:
    for i in range(N):
        fig = plt.figure(figsize=(15, 6))
        plt.title(f"M.S.C N:{i+1}/{N} Trials: {kTrials} time: {time} dt:{dt} r:{r} q:{q}",y=1.05)
        plt.axis("off")
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        trackerX = [[x[i*2] for x in xs] for xs in xs_list]
        trackerY = [[y[i] for y in ys] for ys in ys_list]
        trackerMsX = [[m[i] for m in Ms] for Ms in MsX_list]
        trackerMsY = [[m[i] for m in Ms] for Ms in MsY_list]
        ax1.plot(np.median(trackerX, axis=0), label=f'True State {i+1}', color='blue')
        ax1.plot(np.median(trackerY, axis=0), label=f'Observation {i+1}', color='orange')
        ax1.plot(np.median(trackerMsY, axis=0), label=f'KF Estimate (Observation) {i+1}', color='green')
        ax1.set_title("Median Graph")
        ax1.set_xlabel("Time Steps")
        ax1.legend()


        ax2.set_title("State MSE over Time")
        ax2.plot([r]*len(xs_list[0]), label=f'r={r:.1f}', color='red', linestyle='--')
        ax2.set_xlabel("Time Steps")
        ax2.set_ylabel("MSE")
        

        X_List = [[x[i*2] for x in xs] for xs in xs_list]
        MsX_List = [[ms[i*2] for ms in MsX] for MsX in MsX_list]
        for k in range(kTrials):
            currX = X_List[k]
            currMsX = MsX_List[k]
            ax2.plot([mean_squared_error(currX[:i+1], currMsX[:i+1]) for i in range(len(currX))])
        ax2.legend()

        ax3.set_title("Measurement MSE over Time")
        ax3.plot([q]*len(ys_list[0]),label=f'q={q:.1f}',color='red',linestyle='--')
        ax3.set_xlabel("Time Steps")
        ax3.set_ylabel("MSE")
        Y_List = [[y[i] for y in ys] for ys in ys_list]
        MsY_List = [[ms[i] for ms in MsY] for MsY in MsY_list]
        for k in range(kTrials):
            currY = Y_List[k]
            currMsY = MsY_List[k]
            ax3.plot([mean_squared_error(currY[:i+1], currMsY[:i+1]) for i in range(len(ys))])
        ax3.legend()

        plt.tight_layout()
        if save:
            plt.savefig(f"_07p3_N{i+1}_mass_spring_chain_median.png")

        plt.show()
    
'''
7.4 Non-Linear System: Lorenz-63
zt = (z1,z2,z3)^T
z1' = 10(z2-z1)
z2' = z1(28-z3)-z2
z3' = z1z2 -(8/3)z3

f(z) = (z1',z2',z3')

h=0.01
k1=f(z_{t-1})
k2=f(z_{t-1}+(h/2)k1)
k3=f(z_{t-1}+(h/2)k2)
k4=f(z_{t-1}+(h/2)k3)

z_t = RK4(z_{t-1},h)+ N(0,0.02^2(I_3))
= z_{t-1}+(h/6)(k1+2k2+2k3+k4) + N(0,0.02^2(I_3))

y_t = .5(z1^2+z2^2) +.7z3 + N(0,r)
'''


def fourthOrderRungeKutta(time,h=.01,r=1):
    def f(z):
        z1_dot = 10*(z[1] - z[0])
        z2_dot = z[0] * (28 - z[2]) - z[1]
        z3_dot = z[0] * z[1] - (8/3) * z[2]
        return np.array([z1_dot,z2_dot,z3_dot])
    
    def RK4_step(z):
        k1 = f(z)
        k2 = f(z + (h/2)*k1)
        k3 = f(z + (h/2)*k2)
        k4 = f(z + h*k3)
        return z + (h / 6) * (k1+2*k2+2*k3+k4)
    
    def nextUpdate(z):
        z = RK4_step(z)
        noise = np.random.multivariate_normal(mean=np.zeros(3), cov=(0.02**2) * np.eye(3))
        return z + noise
    
    def measurement(z):
        y = .5 * (z[0]**2 + z[1]**2) + .7*z[2]
        noise = np.random.normal(0,r**2)
        return y + noise
    
    z = np.zeros(3)
    xs=[]
    ys=[]
    for _ in range(time):
        z = nextUpdate(z)
        xs.append(z)
        ys.append(measurement(z))
    return xs,ys
    
np.random.seed(42)
time=1000
kTrials=1
h=0.01
r=.5

xs_list=[]
ys_list=[]
MsX_list=[]
MsY_list=[]
for trial in range(kTrials):
    xs,ys = fourthOrderRungeKutta(time,h=h,r=r)

    x = np.zeros(3)
    P = np.eye(3) * 10
    F = np.eye(3)  # State transition matrix......
    Q = (0.02**2)*np.eye(3)
    H = np.array([[.5,.5,.7]])  # observation transition matrix.......
    R = r**2
    kf = KalmanFilter(x,P,F,H,Q,R)
    
    Ms, Covs = kf.batch_filter(ys)

    #MsX = [ele + np.random.normal(0, q) for ele in Ms]
    MsY = [.5 * (ele[0]**2 + ele[1]**2) + .7*ele[2] + np.random.normal(0, r**2) for ele in Ms]

    xs_list.append(xs)
    ys_list.append(ys)
    #MsX_list.append(MsX)
    MsY_list.append(MsY)
    
if kTrials>0:
    fig = plt.figure(figsize=(15, 6))
    plt.axis('off')
    ax1 = fig.add_subplot(111)
    for t in range(kTrials):
        curr_x = [.5 * (x[0]**2 + x[1]**2) + .7*x[2] for x in xs_list[t]]
        ax1.plot(ys_list[t],label=f'Observations #{t+1}')
        ax1.plot(curr_x,label=f'Actual State #{t+1}')
        ax1.plot(MsY_list[t],label=f'Kalman predictions #{t+1}')
    ax1.legend()
    
    plt.show()