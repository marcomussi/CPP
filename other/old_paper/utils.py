import numpy as np

def generate_curves(num_curves, num_actions=10, base_variation=2, min_start=0.6, max_start=1.0):
    """Generates demand curves"""
    L = base_variation / num_actions
    demands = np.zeros((num_curves, num_actions))
    demands[:, 0] = np.random.uniform(min_start, max_start, num_curves)
    demands[:, 1:] = np.random.uniform(0, -L, (num_curves, num_actions-1))
    demands = np.maximum(np.cumsum(demands, axis=1), 0)
    return demands

def generate_graph(num_products, num_clusters):
    """Generates random clusters"""
    trainers = np.random.choice(num_products, num_clusters, replace=False)
    clusters_dict = {trainers[i] : [] for i in range(len(trainers))}
    to_assign = []
    for i in range(num_products):
        if i not in list(clusters_dict.keys()):
            to_assign.append(i)
    assignments = np.random.choice(list(clusters_dict.keys()), num_products - num_clusters)
    for i in range(len(to_assign)):
        clusters_dict[assignments[i]].append(to_assign[i])
    return clusters_dict

def kernel_rbf(a, b, L): 
    """Radial Basis Function Kernel. Lower "L" means that we are considering smoother functions, acting as a Lipschitz constant"""
    output = -1 * np.ones((a.shape[0], b.shape[0]))
    for i in range(a.shape[0]):
        for j in range(b.shape[0]):
            output[i, j] = np.power(np.linalg.norm(a[i, :] - b[j, :], 2), 2)
    return np.exp(- L * output)