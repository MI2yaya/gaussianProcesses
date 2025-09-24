import numpy as np
import sympy as sp
class ExtendedKalmanFilter:
    def __init__(self, x, P, F, H, Q, R):
        self.x = x  # state mean
        self.P = P  # state covariance
        self.Q = Q  # process noise covariance

        self.F=F # state transition function
        self.H=H # observation function

        self.R = R  # measurement noise covariance



    def jacobian(self, func, x):
        x_symbols = sp.symbols(f'x0:{len(x)}')
        func_sym = func(np.array(x_symbols, dtype=object))
        func_sym = sp.Matrix(func_sym)
        jacobian_matrix = func_sym.jacobian(x_symbols)
        jacobian_func = sp.lambdify(x_symbols, jacobian_matrix, 'numpy')
        return jacobian_func
    
    def predict(self):
        # Predict the next state
        self.x = self.F(self.x)
        F_jacobian = self.jacobian(self.F, self.x)(*self.x) #lambdifed, must use *x
        self.P = F_jacobian @ self.P @ F_jacobian.T + self.Q  
        return self.x, self.P

    def update(self, z):
        # Update the state with measurement z
        H_jacobian = self.jacobian(self.H, self.x)(*self.x)
        y = z - self.H(self.x)
        S = H_jacobian @ self.P @ H_jacobian.T + self.R  # Innovation covariance
        K = self.P @ H_jacobian.T @ np.linalg.inv(S)  # Kalman gain
        self.x = self.x + K @ y  # Update state estimate
        self.P = (np.eye(self.P.shape[0]) - K @ H_jacobian) @ self.P  # Update covariance
        return self.x, self.P
    
    def batch_filter(self, zs):
        estimates = []
        covars = []
        uncertainties=[]
        for z in zs:
            self.x, self.P = self.predict()
            self.x, self.P = self.update(z)
            uncertainties.append(self.return_uncertainty())
            estimates.append(self.x)
            covars.append(self.P)
        return estimates,covars,uncertainties

    def return_uncertainty(self):
        return np.sqrt(np.diag(self.P))