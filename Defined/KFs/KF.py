import numpy as np

class KalmanFilter:
    def __init__(self, x, P, F, H, Q, R):
        self.x = x  # state mean
        self.P = P  # state covariance
        self.Q = Q  # process noise covariance

        self.F=F # state transition matrix
        self.H=H # observation matrix

        self.R = R  # measurement noise covariance
    
    def predict(self):
        # Predict the next state
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q  # Update covariance
        return self.x, self.P
    
    def update(self, z):
        # Update the state with measurement z
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R  # Innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        self.x = self.x + K @ y  # Update state estimate
        self.P = (np.eye(self.P.shape[0]) - K @ self.H) @ self.P  # Update covariance
        return self.x, self.P
    
    def batch_filter(self, zs):
        estimates = []
        covars = []
        for z in zs:
            self.x, self.P = self.predict()
            self.x, self.P = self.update(z)
            estimates.append(self.x)
            covars.append(self.P)
        return estimates,covars
    
    def rts_smoother(self, measurements, covariances):

        return(measurements, covariances)  # Placeholder for RTS smoother implementation