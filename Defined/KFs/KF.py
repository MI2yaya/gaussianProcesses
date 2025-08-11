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
    
    def rts_smoother(self, Xs, Ps):
        n, dim_x = Xs.shape
        
        K = np.zeros((n,dim_x, dim_x))
        x, P = Xs.copy(), Ps.copy()

        for k in range(n-2,-1,-1):
            P_pred = np.dot(self.F, P[k]).dot(self.F.T) + self.Q

            K[k]  = np.dot(P[k], self.F.T).dot(np.linalg.inv(P_pred))
            x[k] += np.dot(K[k], x[k+1] - np.dot(self.F, x[k]))
            P[k] += np.dot(K[k], P[k+1] - P_pred).dot(K[k].T)
        return (x, P)
            