
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Defined.Helpers.plotting import plotMSE, plotHist 
from Defined.KFs.KF import KalmanFilter
from Defined.KFs.UKF import UnscentedKalmanFilter
from PIL import Image
import time as Time

kTrials=0
time=1000
np.random.seed(42)
save=False
r_std=5
q_std=5
'''
7.1 Scalar random walk
'''
def scalarRandomWalk(trials=10, r=1, q=1):
    x_initial = np.random.normal(0, q)
    xs = [x_initial]
    ys = [x_initial+np.random.normal(0,r)]
    x=x_initial
    for _ in range(trials):
        w = np.random.normal(0, q)
        x= x + w
        xs.append(x)
        y = x + np.random.normal(0, r)
        ys.append(y)
    return xs,ys

stateErrors = []
measurementErrors = []
xs_list=[]
ys_list=[]
Ms_list=[]
for trial in range(kTrials):
    xs, ys = scalarRandomWalk(trials=time,r=r_std,q=q_std)
    x = np.array([ys[0]])
    P = np.array([[100]])  # Initial state covariance
    F = np.array([[1.]])  # State transition matrix
    Q = np.array([[q_std**2]])  # Process noise covariance
    H = np.array([[1.]])  # Observation matrix
    R = np.array([[r_std**2]])  # Measurement noise covariance


    kf = KalmanFilter(x,P,F,H,Q,R)



    Ms, Covs= kf.batch_filter(ys)
    Ms, Covs= kf.rts_smoother(np.array(Ms),np.array(Covs))



    steady_start = int(0.2 * len(xs))
    stateErrors.append(mean_squared_error(xs[steady_start:], Ms[steady_start:]))
    measurementErrors.append(mean_squared_error(ys[steady_start:], Ms[steady_start:]))
    xs_list.append(xs)
    ys_list.append(ys)
    Ms_list.append(Ms)


if kTrials>0:
    plotMSE(xs_list, ys_list, Ms_list, Ms_list, r_std, q_std, cov_ex=[cov[0][0] for cov in Covs], save=save, title=f"S.R.W, Trials:{kTrials},time:{time},r_std:{r_std},q_std:{q_std}",name="_07p1a_median_scalar_random_walk.png")
    plotHist(stateErrors,measurementErrors, r_std, q_std, time, kTrials, save=save, title=f"S.R.W, Trials:{kTrials},time:{time},r_std:{r_std},q_std:{q_std}",name="_07p1b_scalar_random_walk_errors.png")


'''
7.2 Constant Velocity Model
x = [px,vx,py,vy] #generalize into N targets
'''
np.random.seed(42)
kTrials=0
time=1000
r_std=5
q_std=5
dt=1
trackers=2
save=False

def constantVelocityModel(trials=10, dt=1, r=1, q=1,trackers=1):
    x_initial = np.random.multivariate_normal(np.zeros(4*trackers), np.eye(4*trackers))
    xs = [x_initial]
    ys = [np.array([x_initial[i] for i in range(0, 4*trackers, 2)])]
    x = x_initial
    for _ in range(trials):
        w = np.random.multivariate_normal(np.zeros(4*trackers), q**2 * np.eye(4*trackers))
        A = np.eye(4*trackers)
        for i in range(0,trackers*4,2):
            A[i][i+1]=dt
        
        x = A @ x + w
        xs.append(x)
        y = np.random.multivariate_normal(np.zeros(2*trackers), r**2 * np.eye(2*trackers))
        
        H= np.zeros((2*trackers, 4*trackers))
        for i in range(0,trackers*2,1):
            H[i][2*i]=1
        
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
    xs, ys = constantVelocityModel(trials=time,dt=dt,r=r_std,q=q_std,trackers=trackers)
    x = np.zeros(4*trackers)  
    P = np.eye(4*trackers)
    
    F = np.eye(4*trackers)
    for i in range(0, trackers*4, 2):
        F[i][i+1] = dt
    
    H = np.zeros((2*trackers, 4*trackers))
    for i in range(0, trackers*2, 1):
        H[i][2*i] = 1
    
    R = np.eye(2*trackers) * r_std**2
    Q = np.eye(4*trackers) * q_std**2 
    kf = KalmanFilter(x, P, F, H, Q, R)

    Ms, Covs = kf.batch_filter(ys)

    MsX = [ele for ele in Ms]
    MsY = [H @ ele for ele in Ms]

    stateErrorsX.append(mean_squared_error([x[0] for x in xs], [m[0] for m in MsX]))
    stateErrorsY.append(mean_squared_error([x[2] for x in xs], [m[2] for m in MsX]))
    measurementErrorsX.append(mean_squared_error([y[0] for y in ys], [m[0] for m in MsY]))
    measurementErrorsY.append(mean_squared_error([y[1] for y in ys], [m[1] for m in MsY]))
    xs_list.append(xs)
    ys_list.append(ys)
    MsX_list.append(MsX)
    MsY_list.append(MsY)


#xs_list [trial][time_step][state]
if kTrials>0:
    for tracker in range(trackers):
        #X-pos plotting for tracker
        trackerX = [[x[0*tracker*4] for x in xs] for xs in xs_list]
        trackerY = [[y[0*tracker*2] for y in ys] for ys in ys_list]
        trackerMsX = [[m[0*tracker*4] for m in Ms] for Ms in MsX_list]
        trackerMsY = [[m[0*tracker*2] for m in Ms] for Ms in MsY_list]
        plotMSE(trackerX, trackerY, trackerMsX, trackerMsY, cov_ex=[cov[0][0] for cov in Covs],r=r_std, q=q_std, save=save, name=f"_07p2a_tracker{tracker}_constant_velocity_model.png",title=f"C.V.M-X Tracker #{tracker+1}/{trackers} Trials:{kTrials},time:{time},r_std:{r_std},q_std:{q_std}")
        
        #Y-pos plotting for tracker
        trackerX = [[x[2+tracker*4] for x in xs] for xs in xs_list]
        trackerY = [[y[1+tracker*2] for y in ys] for ys in ys_list]
        trackerMsX = [[m[2+tracker*4] for m in Ms] for Ms in MsX_list]
        trackerMsY = [[m[1+tracker*2] for m in Ms] for Ms in MsY_list]
        plotMSE(trackerX, trackerY,trackerMsX, trackerMsY, r=r_std, q=q_std, cov_ex=[cov[2][2] for cov in Covs], save=save, name=f"_07p2b_tracker{tracker}_constant_velocity_model.png",title=f"C.V.M-Y Tracker #{tracker+1}/{trackers} Trials:{kTrials},time:{time},r_std:{r_std},q_std:{q_std}")
        
        
        plotHist(stateErrorsX, measurementErrorsX, r_std, q_std, time, kTrials, save=save, name=f"_07p2c_tracker{tracker}_constant_velocity_model.png",title=f"C.V.M X Errors Tracker #{tracker+1}/{trackers}")
        plotHist(stateErrorsY, measurementErrorsY, r_std, q_std, time, kTrials, save=save, name=f"_07p2d_tracker{tracker}_constant_velocity_model.png",title=f"C.V.M Y Errors Tracker #{tracker+1}/{trackers}")
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
kTrials=1
time=1000
r_std=10
q_std=10
dt=.1
Ns=[3]
trial_times=[]

save=False

def massSpringChain(N=3, trials=10, dt=.1, r=1, q=1):
    x_initial = np.random.multivariate_normal(np.zeros(2 * N), np.eye(2 * N))
    xs = [x_initial]
    ys = [np.array([x_initial[i] for i in range(0, 2 * N, 2)])]
    x = x_initial
    for _ in range(trials):
        w = np.random.multivariate_normal(np.zeros(2 * N), q**2 * np.eye(2 * N))
        A = np.eye(2 * N)
        for i in range(N):
            if i > 0:
                A[2*i, 2*i-1] = -dt
            if i < N-1:
                A[2*i+1, 2*i+2] = dt
        x = A @ x + w
        xs.append(x)
        y = np.random.multivariate_normal(np.zeros(N), r**2 * np.eye(N))
        H = np.zeros((N, 2 * N))
        for i in range(N):
            H[i, 2*i] = 1
        y_observed = H @ x + y
        ys.append(y_observed)
    return xs, ys
for N in Ns:
    stateErrors = []
    measurementErrors = []
    xs_list=[]
    ys_list=[]
    MsX_list=[]
    MsY_list=[]
    start_time = Time.time()
    for trial in range(kTrials):
        xs, ys = massSpringChain(N=N,trials=time,q=q_std,r=r_std,dt=dt)
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
        R = np.eye(N) * r_std**2
        Q = np.eye(2 * N) * q_std**2

        kf = KalmanFilter(x, P, F, H, Q, R)


        Ms, Covs= kf.batch_filter(ys)

        MsX = [ele for ele in Ms]
        MsY = [H @ ele for ele in Ms]

        xs_list.append(xs)
        ys_list.append(ys)
        MsX_list.append(MsX)
        MsY_list.append(MsY)


        for i in range(N):
            if trial==0:
                stateErrors.append([])
                measurementErrors.append([])
            stateErrors[i].append(mean_squared_error([x[i*2] for x in xs], [m[i*2] for m in MsX]))
            measurementErrors[i].append(mean_squared_error([y[i] for y in ys], [m[i] for m in MsY]))
    totalTime = Time.time() - start_time
    trial_times.append(totalTime)
    print(f'Time Elapsed: {totalTime:.4f}, N: {N}')
fig=plt.figure(figsize=(10,5))
plt.title(f'M.S.C Times for select N Trials: {kTrials} time: {time} dt:{dt} r_std:{r_std} q_std:{q_std}',y=1.05)
plt.axis('off')
axis = fig.add_subplot(111)
axis.plot(Ns,trial_times)
axis.set_xlabel("N")
axis.set_ylabel("time (s)")
if save:
    plt.savefig('_07p3c_N_mass_spring_chain_times.png')
plt.show()
if kTrials>0:
    for i in range(N):
        fig = plt.figure(figsize=(15, 6))
        plt.title(f"M.S.C N:{i+1}/{N} Trials: {kTrials} time: {time} dt:{dt} r_std:{r_std} q_std:{q_std}",y=1.05)
        plt.axis("off")
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        trackerX = [[x[i*2] for x in xs] for xs in xs_list]
        trackerY = [[y[i] for y in ys] for ys in ys_list]
        trackerMsX = [[m[i] for m in Ms] for Ms in MsX_list]
        trackerMsY = [[m[i] for m in Ms] for Ms in MsY_list]
        ax1.plot(np.median(trackerX, axis=0), label=f'True State #{i+1}', color='blue')
        ax1.plot(np.median(trackerY, axis=0), label=f'Observation #{i+1}', color='orange')
        ax1.plot(np.median(trackerMsY, axis=0), label=f'KF Estimate (Observation) #{i+1}', color='green')
        ax1.set_title("Median Graph")
        ax1.set_xlabel("Time Steps")
        ax1.legend()


        ax2.set_title("State MSE over Time")
        ax2.plot([r_std**2]*len(xs_list[0]), label=f'r_var={r_std**2:.1f}', color='red', linestyle='--')
        ax2.plot([cov[i*2][i*2] for cov in Covs],label='cov',color='blue',linestyle='--')
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
        ax3.plot([q_std**2]*len(ys_list[0]),label=f'q_var={q_std**2:.1f}',color='red',linestyle='--')
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

        plotHist(stateErrors[i],measurementErrors[i],r_std,q_std,time,kTrials,save,f"_07p3_N{i+1}_mass_spring_chain_errors.png",f"M.S.C N:{i+1}/{N}")
    if save:
        img1_path = f"_07p3_N1_mass_spring_chain_median.png"
        img2_path = f"_07p3_N2_mass_spring_chain_median.png"
        img3_path = f"_07p3_N3_mass_spring_chain_median.png"
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)
        img3 = Image.open(img3_path)
        combined = Image.new('RGB', (img1.width, img1.height + img2.height + img3.height))
        combined.paste(img1, (0, 0))
        combined.paste(img2, (0, img1.height))
        combined.paste(img3, (0, img1.height+img2.height))
        combined.save(f"_07p3a_N3_mass_spring_chain_median_combined.png")
        img1.close()
        img2.close()
        img3.close()
        os.remove(img1_path)
        os.remove(img2_path)
        os.remove(img3_path)
        
        img1_path = f"_07p3_N1_mass_spring_chain_errors.png"
        img2_path = f"_07p3_N2_mass_spring_chain_errors.png"
        img3_path = f"_07p3_N3_mass_spring_chain_errors.png"
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)
        img3 = Image.open(img3_path)
        combined = Image.new('RGB', (img1.width, img1.height + img2.height + img3.height))
        combined.paste(img1, (0, 0))
        combined.paste(img2, (0, img1.height))
        combined.paste(img3, (0, img1.height+img2.height))
        combined.save(f"_07p3b_N3_mass_spring_chain_error_combined.png")
        img1.close()
        img2.close()
        img3.close()
        os.remove(img1_path)
        os.remove(img2_path)
        os.remove(img3_path)
    
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
        noise = np.random.normal(0,r)
        return y + noise
    
    z = np.zeros(3)
    xs=[]
    ys=[]
    for _ in range(time):
        z = nextUpdate(z)
        xs.append(z)
        ys.append(measurement(z))
    return xs,ys

def fx(x):
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
    

    return RK4_step(x)
def hx(x):
     return np.array([.5*(x[0]**2 + x[1]**2) + .7*x[2]])
 
np.random.seed(42)
time=100
kTrials=5
h=0.01
r_std=.5
save=False

xs_list=[]
ys_list=[]
kf_MsY_list=[]
ukf_MsY_list=[]
kf_measurementErrors = []
ukf_measurementErrors = []
kf_errors=[]
ukf_errors=[]
for trial in range(kTrials):
    xs,ys = fourthOrderRungeKutta(time,h=h,r=r_std)

    x = np.zeros(3)
    P = np.eye(3) * 10
    F = np.eye(3)  # State transition matrix......
    Q = (0.02**2)*np.eye(3)
    H = np.array([[.5,.5,.7]])  # observation transition matrix.......
    R = np.array([[r_std**2]])
    kf = KalmanFilter(x,P,F,H,Q,R)
    ukf = UnscentedKalmanFilter(x,P,fx,hx,Q,R,n=3)
    
    kf_Ms, kf_Covs = kf.batch_filter(ys)
    ukf_Ms, ukf_Covs = ukf.batch_filter(ys)
    kf_MsY = [hx(Ms) for Ms in kf_Ms]
    ukf_MsY = [hx(Ms) for Ms in ukf_Ms]


    xs_list.append(xs)
    ys_list.append(ys)
    kf_MsY_list.append(kf_MsY)
    ukf_MsY_list.append(ukf_MsY)
    
    kf_errors.append(mean_squared_error(ys, kf_MsY))
    ukf_errors.append(mean_squared_error(ys, ukf_MsY))


    
if kTrials>0:
    fig = plt.figure(figsize=(15, 6))
    plt.axis('off')
    plt.title(f"Lorenz63 KF vs UKF Trials:{kTrials} time:{time} r_std:{r_std} h:{h} N:{3}")
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    conv_x = np.median([[hx(x) for x in xs] for xs in xs_list],axis=0)
    conv_y = np.median(ys_list,axis=0)
    ax1.set_title("Median KF")
    ax2.set_title("Median UKF")
    ax3.set_title("KF MSE")
    ax4.set_title("UKF MSE")
    
    ax1.plot(conv_y,label=f'Observations')
    ax1.plot(conv_x,label=f'Actual State')
    ax1.plot(np.median(kf_MsY_list,axis=0),label=f'KF predictions')
    ax1.set_xlabel("Time")
    
    ax2.plot(conv_y,label=f'Observations')
    ax2.plot(conv_x,label=f'Actual State')
    ax2.plot(np.median(ukf_MsY_list,axis=0),label=f'UKF Prediction')
    ax2.set_xlabel("Time")
    for t in range(kTrials):
        curr_y = ys_list[t]
        curr_kf_msy = kf_MsY_list[t]
        curr_ukf_msy = ukf_MsY_list[t]
        
        ax3.plot([mean_squared_error(curr_y[:i+1], curr_kf_msy[:i+1]) for i in range(len(ys))])
        ax4.plot([mean_squared_error(curr_y[:i+1], curr_ukf_msy[:i+1]) for i in range(len(ys))])
    
    ax3.plot([.5**2]*len(ys),label='Min Error',linestyle='--')
    ax3.set_ylabel("MSE")
    ax3.set_xlabel("Time")
    
    ax4.plot([.5**2]*len(ys),label='Min Error',linestyle='--')
    ax4.set_ylabel("MSE")
    ax4.set_xlabel("Time")
    
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    plt.tight_layout()
    if save:
        plt.savefig('_07p4a_Lorenz63.png')
    plt.show()
    plotHist(kf_errors, ukf_errors, r_std, r_std, time, kTrials, save=save, title=f"KF vs UKF",t1='KF',t2="UKF",name='_07p4b_Lorenz63.png')

        