import numpy as np

# Credit: https://filippomb.github.io/python-time-series-handbook/notebooks/07/kalman-filter.html
# Additional Credit: https://www.geeksforgeeks.org/python/kalman-filter-in-python/

class KalmanFilter:
    def __init__(self, dt, x0):
        self.dt = dt
        
        self.F = np.array([
            [1,0,dt,0],
            [0,1,0,dt],
            [0,0,1,0],
            [0,0,0,1]])
        self.B = np.zeros((4,4))
        self.H = np.array([
            [1,0,0,0],
            [0,1,0,0]])
        self.Q = np.eye(4) * 0.03
        self.R = np.eye(2) * 5
        self.x = x0
        self.P = np.eye(4)
    
    def predict(self, u=None):
        if u is None:
            u = np.zeros((self.B.shape[1],1))
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
        return self.x
    
    def update(self, z):
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        y = z - np.dot(self.H, self.x)
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.P.shape[0])
        self.P = np.dot(I - np.dot(K, self.H), self.P)
        return self.x
    