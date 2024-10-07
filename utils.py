import numpy as np


def generate_curves(num_curves, num_actions=10, base_variation=2):
    """
    Generates demand curves
    """

    L = base_variation / num_actions
    
    demands = np.zeros((num_curves, num_actions))
    demands[:, 0] = np.random.uniform(0.6, 1.0, num_curves)
    demands[:, 1:] = np.random.uniform(0, -L, (num_curves, num_actions-1))
    demands = np.maximum(np.cumsum(demands, axis=1), 0)

    return demands


def generate_user_ranges(num_users, min_range, max_range, low_to_high_mix=0.5):
    """
    Generates user ranges for low and high selling products
    """
    
    user_ranges = np.zeros((num_users, 2), dtype=int)
    low_mask = np.random.uniform(0, 1, num_users) > low_to_high_mix
    user_ranges[low_mask, 0] = min_range
    user_ranges[np.logical_not(low_mask), 1] = max_range
    user_ranges[low_mask, 1] = np.random.randint(low=min_range, high=max_range, size=np.sum(low_mask))
    user_ranges[np.logical_not(low_mask), 0] = np.random.randint(low=min_range, high=max_range, size=num_users-np.sum(low_mask))
    
    return user_ranges


def kernel_rbf(a, b, L): 
    """
    Radial Basis Function Kernel 
    Lower "L" means that we are considering smoother functions, acting as a Lipschitz constant
    """
    
    output = -1 * np.ones((a.shape[0], b.shape[0]))
    
    for i in range(a.shape[0]):
        for j in range(b.shape[0]):
            output[i, j] = np.power(np.linalg.norm(a[i, :] - b[j, :], 2), 2)
    
    return np.exp(- L * output)
