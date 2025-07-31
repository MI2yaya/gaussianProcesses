import numpy as np

class KalmanFilter:
    def __init__(self, x, P, F, H, Q, R):
        self.x = x  # state mean
        self.P = P  # state covariance
        self.Q = Q  # process noise covariance

        self.F=F # state transition matrix
        self.H=H # observation matrix

        self.R = R  # measurement noise covariance
    


    def predict(self, u):
        # Predict the next state
        self.x = self.F @ self.x + u
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
    
    def batch_filter(self, measurements):
        estimates = []
        for z in measurements:
            self.x, self.P = self.predict(z)
            self.x, self.P = self.update(z)
            estimates.append(self.x)
        return estimates
    
    def rts_smoother(self, measurements):
        n = len(measurements)
        smoothed_estimates = np.zeros((n, self.x.shape[0]))
        smoothed_estimates[-1] = self.x
        P = self.P.copy()
        for i in range(n - 2, -1, -1):
            F = self.F
            H = self.H
            K = P @ H.T @ np.linalg.inv(H @ P @ H.T + self.R)
            smoothed_estimates[i] = self.x + K @ (smoothed_estimates[i + 1] - F @ self.x)
            P = P + K @ (P - F @ P) @ K.T
        return smoothed_estimates