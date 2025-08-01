import numpy as np

class UnscentedKalmanFilter:
    def __init__(self,x,P,fx,hx,Q,R,n,alpha=1e-3,beta=2,kappa=0.0):
        self.x = x
        self.P = P
        self.Q = Q
        
        self.fx = fx
        self.hx = hx
        
        self.R = R
        
        self.n=n
        self.alpha=alpha
        self.beta=beta
        self.kappa=kappa
        self.lam = alpha**2 * (self.n + self.kappa) - self.n
        
        
        self.Wm = np.full(2 * self.n + 1, 1 / (2 * (self.n + self.lam)))
        self.Wc = np.copy(self.Wm)
        self.Wm[0] = self.lam / (self.n + self.lam)
        self.Wc[0] = self.lam / (self.n + self.lam) + (1 - alpha**2 + beta)
        
    
    def generateSigmaPoints(self,x,P):
        #MerweScaledSigmaPoints
        sigma_points = np.zeros((2 * self.n + 1, self.n))
        U = np.linalg.cholesky((self.n + self.lam) * P)

        sigma_points[0] = x
        for i in range(self.n):
            sigma_points[i + 1]     = x + U[i]
            sigma_points[self.n + i + 1] = x - U[i]
        return sigma_points
        
    
    def predict(self):
        sigmas = self.generateSigmaPoints(self.x,self.P)
        sigmas_f = np.array([self.fx(sigma) for sigma in sigmas])
        
        x_pred = np.dot(self.Wm,sigmas_f)
        P_pred = np.zeros((self.n, self.n))
        
        for i in range(2 * self.n + 1):
            y = sigmas_f[i] - x_pred
            P_pred += self.Wc[i] * np.outer(y, y)
        P_pred += self.Q
        
        self.x = x_pred
        self.P = P_pred
        self.sigmas_f = sigmas_f
        return self.x,self.P
        
    def update(self,z):
        sigmas_h = np.array([self.hx(s) for s in self.sigmas_f])
        
        z_pred = np.dot(self.Wm,sigmas_h)
        
        Pz = np.zeros((z_pred.size, z_pred.size))
        for i in range(2*self.n+1):
            dz = sigmas_h[i] - z_pred
            Pz += self.Wc[i] * np.outer(dz,dz)
        Pz += self.R
    

        Pxz = np.zeros((self.n, z_pred.size))
        for i in range(2*self.n+1):
            dx = self.sigmas_f[i] - self.x
            dz = sigmas_h[i] - z_pred
            Pxz += self.Wc[i] * np.outer(dx,dz)
        
        K = Pxz @ np.linalg.inv(Pz)
        self.x = self.x + K @(z-z_pred)
        self.P = self.P - K @ Pz @ K.T
        return self.x,self.P
        
    def batch_filter(self,zs):
        estimates = []
        covars = []
        for z in zs:
            self.predict()
            self.update(z)
            estimates.append(self.x.copy())
            covars.append(self.P.copy())
        return estimates,covars
    
    def rts_smoother(self, measurements, covariances):

        return(measurements, covariances)  # Placeholder for RTS smoother implementation