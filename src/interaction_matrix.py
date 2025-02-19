import torch
import numpy as np

def Sherrington_Kirkpatrick_1D(system_size):

    # Sample a full matrix from a normal distribution
    J = np.zeros((system_size, system_size))

    # Get the indices for the strictly upper triangular part (k=1 excludes the diagonal)
    upper_indices = np.triu_indices(system_size, k=1)

    # Fill the strictly upper triangular part with samples from N(0,1)
    J[upper_indices] = np.random.normal(loc=0, scale=1, size=len(upper_indices[0]))

    return J

def Fully_connected_1D(system_size):

    return np.triu(np.ones((system_size, system_size)), k=1)
