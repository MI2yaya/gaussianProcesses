import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

def plotMSE(xs, ys, MsX, MsY, r, q, save=False, name="smth.png",title="Kalman Filter MSE"):
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    ax1.set_title(title)
    ax1.plot(xs, label='True States', color='blue')
    ax1.plot(ys, label='Observations', color='orange')
    ax1.plot(Ms, label='Kalman Filter Estimate (Observation)', color='green')
    ax1.set_title("State vs Observation vs Estimate")
    ax1.set_xlabel("Time Steps")
    ax1.legend()
    ax2.set_title("State MSE over Time")

    ax2.plot([mean_squared_error(xs[:i+1], MsX[:i+1]) for i in range(len(xs))], label='MSE', color='blue')
    ax2.plot([r]*len(xs), label='Process Noise', color='red', linestyle='--')
    ax2.set_xlabel("Time Steps")
    ax2.set_ylabel("MSE")
    ax2.legend()
    ax3.set_title("Measurement MSE over Time")


    ax3.plot([mean_squared_error(ys[:i+1], MsY[:i+1]) for i in range(len(ys))], label='MSE', color='blue')
    ax3.plot([q]*len(ys), label='Measurement Noise', color='red', linestyle='--')
    ax3.set_xlabel("Time Steps")
    ax3.set_ylabel("MSE")
    ax3.legend()
    plt.tight_layout()
    if save:
        plt.savefig(name)
    plt.show()

def plotHist(stateErrors,measurementErrors, r, q, time,kTrials,save=False,name="smth.png",title="Kalman Filter Error Histograms"):
    fig = plt.figure(figsize=(15, 6))
    plt.axis("off")
    plt.title(f"{title}; time={time}, trials={kTrials}")
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    ax1.hist(stateErrors, bins=30, alpha=0.7)
    ax1.set_title("State Errors")
    ax1.set_xlabel("MSE")
    ax1.set_ylabel("Frequency")
    ax1.text(0.5, 0.9, f"Mean: {np.mean(stateErrors):.3f}\nMedian: {np.median(stateErrors):.3f}",
                transform=ax1.transAxes, fontsize=12, verticalalignment='top')
    ax2.hist(measurementErrors, bins=30, alpha=0.7)
    ax2.set_title("Measurement Errors")
    ax2.set_xlabel("MSE")
    ax2.set_ylabel("Frequency")
    ax2.text(0.5, 0.9, f"Mean: {np.mean(measurementErrors):.3f}\nMedian: {np.median(measurementErrors):.3f}",
                transform=ax2.transAxes, fontsize=12, verticalalignment='top')
    ax3.plot(stateErrors, label='State Errors', color='blue')
    ax3.set_title("State Errors Over Trials")
    ax3.set_xlabel("Trial")
    ax3.set_ylabel("MSE")
    ax3.plot([r]*kTrials, label='Process Noise', color='red')
    ax3.legend()
    ax4.plot(measurementErrors, label='Measurement Errors', color='orange')
    ax4.set_title("Measurement Errors Over Trials")
    ax4.set_xlabel("Trial")
    ax4.set_ylabel("MSE")
    ax4.plot([q]*kTrials, label='Measurement Noise', color='green')
    ax4.legend()
    plt.tight_layout()
    if save:
        plt.savefig(name)
    plt.show()