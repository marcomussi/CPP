import numpy as np

from utils import kernel_rbf, incr_inv



class GaussianProcessRegressor: 

    
    def __init__(self, kernel_L, sigma_sq_process, input_dim):
        
        self.kernel_L = kernel_L
        self.sigma_sq_process = sigma_sq_process  
        self.input_dim = input_dim
        self.x_vect = None
        self.y_vect = None
            
    
    def update_data(self, x, y):
        
        if self.x_vect is None:
            
            self.x_vect = np.array([x]).reshape(1, self.input_dim)
            self.y_vect = np.array([y]).reshape(1, 1)
            
            self.K_matrix = np.array([[kernel_rbf(np.array([x]).reshape(1, self.input_dim), 
                                      np.array([x]).reshape(1, self.input_dim), 
                                      self.kernel_L) + self.sigma_sq_process]])
            self.K_matrix_inv = np.linalg.inv(self.K_matrix)
        
        else:

            new_K = kernel_rbf(self.x_vect, np.array([x]).reshape(1, self.input_dim), self.kernel_L)
            
            self.x_vect = np.vstack((self.x_vect, np.array([x]).reshape(1, self.input_dim)))
            self.y_vect = np.vstack((self.y_vect, np.array([y]).reshape(1, 1)))

            self.K_matrix_inv = incr_inv(
                self.K_matrix_inv, 
                new_K.reshape(-1, 1), 
                new_K.reshape(1, -1),
                np.array([[kernel_rbf(np.array([x]).reshape(1, self.input_dim), 
                                      np.array([x]).reshape(1, self.input_dim), 
                                      self.kernel_L) + self.sigma_sq_process]])
            )


    def load_data(self, x, y):

        n = x.shape[0]
        self.x_vect = np.array([x]).reshape(n, self.input_dim)
        self.y_vect = np.array([y]).reshape(n, 1)
        
        self.K_matrix = kernel_rbf(self.x_vect, self.x_vect, self.kernel_L) + self.sigma_sq_process * np.eye(n)
        self.K_matrix_inv = np.linalg.inv(self.K_matrix)

    
    def compute(self, x):
        
        assert x.ndim == 2, "compute() function: Error in input dimension"
        assert x.shape[1] == self.input_dim, "compute() function: Error in input dimension"
        
        n = x.shape[0]
        mu = np.zeros(n)
        sigma = np.zeros(n)
        
        for i in range(n):
            
            K_star = kernel_rbf(self.x_vect, x[i, :].reshape(1, self.input_dim), self.kernel_L)
            mu[i] = K_star.T @ self.K_matrix_inv @ self.y_vect
            sigma[i] = kernel_rbf(x[i, :].reshape(1, self.input_dim), x[i, :].reshape(1, self.input_dim), self.kernel_L
                                 ) - K_star.T @ self.K_matrix_inv @ K_star
        
        return mu, sigma
    
    
    
    
class DoubleGaussianProcessRegressor: 
    
    
    def __init__(self, kernel_L, sigma_sq_process, input_dim):
        
        self.kernel_L = kernel_L
        self.sigma_sq_process = sigma_sq_process  
        self.input_dim = input_dim
        self.x_vect = None
        self.y1_vect = None
        self.y2_vect = None

    
    def update_data(self, x, y1, y2):
        
        if self.x_vect is None:
            
            self.x_vect = np.array([x]).reshape(1, self.input_dim)
            self.y1_vect = np.array([y1]).reshape(1, 1)
            self.y2_vect = np.array([y2]).reshape(1, 1)

            self.K_matrix = np.array([[kernel_rbf(x.reshape(1, self.input_dim), x.reshape(1, self.input_dim), self.kernel_L
                                 ) + self.sigma_sq_process]])
            self.K_matrix_inv = np.linalg.inv(self.K_matrix)
        
        else:

            new_K = kernel_rbf(self.x_vect, np.array([x]).reshape(1, self.input_dim), self.kernel_L)
            
            self.x_vect = np.vstack((self.x_vect, np.array([x]).reshape(1, self.input_dim)))
            self.y1_vect = np.vstack((self.y1_vect, np.array([y1]).reshape(1, 1)))
            self.y2_vect = np.vstack((self.y2_vect, np.array([y2]).reshape(1, 1)))
            
            self.K_matrix_inv = incr_inv(
                self.K_matrix_inv, 
                new_K.reshape(-1, 1), 
                new_K.reshape(1, -1),
                np.array([[kernel_rbf(x.reshape(1, self.input_dim), x.reshape(1, self.input_dim), self.kernel_L
                                 ) + self.sigma_sq_process]])
            )
        
        
    def load_data(self, x, y1, y2):

        n = x.shape[0]
        self.x_vect = x.reshape(n, self.input_dim)
        self.y1_vect = y1.reshape(n, 1)
        self.y2_vect = y2.reshape(n, 1)

        self.K_matrix = kernel_rbf(self.x_vect, self.x_vect, self.kernel_L) + self.sigma_sq_process * np.eye(n)
        self.K_matrix_inv = np.linalg.inv(self.K_matrix)


    def compute(self, x):
        
        assert x.ndim == 2, "compute() function: Error in input dimension"
        assert x.shape[1] == self.input_dim, "compute() function: Error in input dimension"
        
        n = x.shape[0]
        
        y1_mu = np.zeros(n)
        y2_mu = np.zeros(n)
        sigma = np.zeros(n)
        
        for i in range(n):
            
            K_star = kernel_rbf(self.x_vect, x[i, :].reshape(1, self.input_dim), self.kernel_L)
            sigma[i] = kernel_rbf(x[i, :].reshape(1, self.input_dim), x[i, :].reshape(1, self.input_dim), self.kernel_L
                                 ) - K_star.T @ self.K_matrix_inv @ K_star
            y1_mu[i] = K_star.T @ self.K_matrix_inv @ self.y1_vect
            y2_mu[i] = K_star.T @ self.K_matrix_inv @ self.y2_vect
            
        return y1_mu, y2_mu, sigma