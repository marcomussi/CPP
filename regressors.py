import numpy as np

from utils import kernel_rbf


class GaussianProcessRegressor: 

    
    def __init__(self, kernel_L, sigma_process, input_dim):
        
        self.kernel_L = kernel_L
        self.sigma_process = sigma_process  
        self.input_dim = input_dim
        self.reset()


    def reset(self):
        
        self.x_vect = None
        self.y_vect = None
            
    
    def update_data(self, x, y):
        
        if self.x_vect is None:
            self.x_vect = np.array([x]).reshape(1, self.input_dim)
            self.y_vect = np.array([y]).reshape(1, 1)
        else:
            self.x_vect = np.vstack((self.x_vect, np.array([x]).reshape(1, self.input_dim)))
            self.y_vect = np.vstack((self.y_vect, np.array([y]).reshape(1, 1)))

    
    def compute(self, x):
        
        assert x.ndim == 2, "compute() function: Error in input dimension"
        assert x.shape[1] == self.input_dim, "compute() function: Error in input dimension"
        n = x.shape[0]
        
        K = kernel_rbf(self.x_vect, self.x_vect, self.kernel_L) + self.sigma_process * np.eye(self.y_vect.shape[0])
        K_inv = np.linalg.inv(K)
        
        self.mu = np.zeros(n)
        self.sigma = np.zeros(n)
        for i in range(n):
            K_star = kernel_rbf(self.x_vect, x[i, :].reshape(1, self.input_dim), self.kernel_L)
            self.mu[i] = K_star.T @ K_inv @ self.y_vect
            self.sigma[i] = 1 - K_star.T @ K_inv @ K_star
        
        return self.mu, self.sigma
