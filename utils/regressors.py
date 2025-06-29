import numpy as np


def kernel_rbf(a, b, L): 
    """Radial Basis Function Kernel. Lower "L" means that we are considering smoother functions"""
    output = -1 * np.ones((a.shape[0], b.shape[0]))
    for i in range(a.shape[0]):
        for j in range(b.shape[0]):
            output[i, j] = np.power(np.linalg.norm(a[i, :] - b[j, :], 2), 2)
    return np.exp(- L * output)


class HeteroscedasticGaussianProcessRegressor: 

    def __init__(self, kernel_L, input_dim=1):
        self.kernel_L = kernel_L
        self.input_dim = input_dim

    def load_data(self, x, y, sigmas):
        n = x.shape[0]
        self.x_vect = np.array([x]).reshape(n, self.input_dim)
        self.y_vect = np.array([y]).reshape(n, 1)
        self.sigmas = np.array([sigmas]).reshape(n,)

    def compute(self, x):
        assert x.ndim == 2, "compute() function: Error in input dimension"
        assert x.shape[1] == self.input_dim, "compute() function: Error in input dimension"
        n = x.shape[0]
        K = kernel_rbf(self.x_vect, self.x_vect, self.kernel_L) + np.diag(self.sigmas)
        K_inv = np.linalg.inv(K)
        mu = np.zeros(n)
        sigma = np.zeros(n)
        for i in range(n):
            K_star = kernel_rbf(self.x_vect, x[i, :].reshape(1, self.input_dim), self.kernel_L)
            mu[i] = K_star.T @ K_inv @ self.y_vect
            sigma[i] = 1 - K_star.T @ K_inv @ K_star
        return mu, sigma


class HeteroscedasticDoubleGaussianProcessRegressor: 

    def __init__(self, kernel_L, input_dim=1):
        self.kernel_L = kernel_L
        self.input_dim = input_dim
        self.x_vect = None
        self.y1_vect = None
        self.y2_vect = None
        self.sigmas = None

    def load_data(self, x, y1, y2, sigmas):
        n = x.shape[0]
        self.x_vect = x.reshape(n, self.input_dim)
        self.y1_vect = y1.reshape(n, 1)
        self.y2_vect = y2.reshape(n, 1)
        self.sigmas = np.array([sigmas]).reshape(n,)

    def compute(self, x):
        assert x.ndim == 2, "compute() function: Error in input dimension"
        assert x.shape[1] == self.input_dim, "compute() function: Error in input dimension"
        n = x.shape[0]
        K = kernel_rbf(self.x_vect, self.x_vect, self.kernel_L) + np.diag(self.sigmas)
        K_inv = np.linalg.inv(K)
        y1_mu = np.zeros(n)
        y2_mu = np.zeros(n)
        sigma = np.zeros(n)
        for i in range(n):
            K_star = kernel_rbf(self.x_vect, x[i, :].reshape(1, self.input_dim), self.kernel_L)
            sigma[i] = 1 - K_star.T @ K_inv @ K_star
            y1_mu[i] = K_star.T @ K_inv @ self.y1_vect
            y2_mu[i] = K_star.T @ K_inv @ self.y2_vect
        return y1_mu, y2_mu, sigma