import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

def plotMSE(xs_list, ys_list, MsX_list, MsY_list, r, q, save=False, name="smth.png",title="Kalman Filter MSE"):
    fig = plt.figure(figsize=(15, 6))
    plt.title(f"{title}",y=1.05)
    plt.axis("off")
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    ax1.plot(np.median(xs_list,axis=0), label='True States', color='blue')
    ax1.plot(np.median(ys_list,axis=0), label='Observations', color='orange')
    ax1.plot(np.median(MsY_list,axis=0), label='KF Estimate (Observation)', color='green')
    ax1.set_title("Median Graph")
    ax1.set_xlabel("Time Steps")
    ax1.legend()
    
    
    ax2.set_title("State MSE over Time")
    ax2.plot([r]*len(xs_list[0]), label=f'r={r:.1f}', color='red', linestyle='--')
    ax2.set_xlabel("Time Steps")
    ax2.set_ylabel("MSE")
    for k in range(len(MsX_list)):
        xs = xs_list[k]
        MsX = MsX_list[k]
        ax2.plot([mean_squared_error(xs[:i+1], MsX[:i+1]) for i in range(len(xs))])
    ax2.legend()
    
    
    ax3.set_title("Measurement MSE over Time")
    ax3.plot([q]*len(ys_list[0]),label=f'q={q:.1f}',color='red',linestyle='--')
    ax3.set_xlabel("Time Steps")
    ax3.set_ylabel("MSE")
    for k in range(len(MsY_list)):
        ys = ys_list[k]
        MsY = MsY_list[k]
        ax3.plot([mean_squared_error(ys[:i+1], MsY[:i+1]) for i in range(len(ys))])
    ax3.legend()
    
    plt.tight_layout()
    if save:
        plt.savefig(name)
  
    plt.show()

def plotHist(stateErrors,measurementErrors, r, q, time,kTrials,save=False,name="smth.png",title="Kalman Filter Error Histograms",t1="State",t2='Measurement'):
    fig = plt.figure(figsize=(15, 6))
    plt.axis("off")
    plt.title(f"{title}; time={time}, trials={kTrials}")
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    ax1.hist(stateErrors, bins=30, alpha=0.7)
    ax1.set_title(f"{t1} Errors")
    ax1.set_xlabel("MSE")
    ax1.set_ylabel("Frequency")
    q1= np.quantile(stateErrors, 0.25)
    q3= np.quantile(stateErrors, 0.75)
    mean = np.mean(stateErrors)
    median = np.median(stateErrors)
    ax1.axvline(q1, color='red', linestyle='--', label='Q1')
    ax1.axvline(mean, color='blue', linestyle='--', label=f'Mean {mean:.3f}')
    ax1.axvline(median, color='orange', linestyle='--', label=f'Median {median:.3f}')
    ax1.axvline(q3, color='green', linestyle='--', label='Q3')
    ax1.legend()
    
    ax2.hist(measurementErrors, bins=30, alpha=0.7)
    ax2.set_title(f"{t2} Errors")
    ax2.set_xlabel("MSE")
    ax2.set_ylabel("Frequency")
    q1 = np.quantile(measurementErrors, 0.25)
    q3 = np.quantile(measurementErrors, 0.75)
    mean = np.mean(measurementErrors)
    median = np.median(measurementErrors)
    ax2.axvline(q1, color='red', linestyle='--', label='Q1')
    ax2.axvline(mean, color='blue', linestyle='--', label=f'Mean {mean:.3f}')
    ax2.axvline(median, color='orange', linestyle='--', label=f'Median {median:.3f}')
    ax2.axvline(q3, color='green', linestyle='--', label='Q3')
    ax2.legend()
    
    ax3.plot(stateErrors, label=f'{t1} Errors', color='blue')
    ax3.set_title(f"{t1} Errors Over Trials")
    ax3.set_xlabel("Trial")
    ax3.set_ylabel("MSE")
    ax3.plot([r]*kTrials, label='Process Noise', color='red')
    ax3.legend()

    ax4.plot(measurementErrors, label=f'{t2} Errors', color='orange')
    ax4.set_title(f"{t2} Errors Over Trials")
    ax4.set_xlabel("Trial")
    ax4.set_ylabel("MSE")
    ax4.plot([q]*kTrials, label='Measurement Noise', color='green')
    ax4.legend()
    plt.tight_layout()
    if save:
        plt.savefig(name)
    plt.show()