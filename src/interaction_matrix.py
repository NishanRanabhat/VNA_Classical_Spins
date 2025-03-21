import torch
import numpy as np


def Sherrington_Kirkpatrick_1D(system_size:int):
    """
    Generate a symmetric interaction matrix for the Sherrington-Kirkpatrick model 
    in one dimension by sampling from a normal distribution.

    This function creates a square matrix of shape (system_size, system_size) 
    where the strictly upper triangular entries (excluding the diagonal) are 
    independently sampled from a normal distribution with mean 0 and standard deviation 1.
    The lower triangular part is implicitly zero (or can be assumed to be symmetric if needed).

    Parameters:
        system_size (int): The number of spins (or nodes) in the system. 
                           This determines the dimensions of the matrix.

    Returns:
        numpy.ndarray: A 2D NumPy array of shape (system_size, system_size) with the upper
                       triangular part filled with normally distributed random values and 
                       zeros elsewhere.
                       
    Example:
        >>> np.random.seed(0)
        >>> matrix = Sherrington_Kirkpatrick_1D(4)
        >>> print(matrix)
        [[ 0.          1.76405235  0.40015721  0.97873798]
         [ 0.          0.          2.2408932   1.86755799]
         [ 0.          0.          0.          0.95008842]
         [ 0.          0.          0.          0.        ]]
    """
    # Create a zero matrix of the desired size
    J = np.zeros((system_size, system_size))
    
    # Get the indices for the strictly upper triangular part (excluding the diagonal)
    upper_indices = np.triu_indices(system_size, k=1)
    
    # Fill the strictly upper triangular part with samples from a normal distribution (mean=0, std=1)
    J[upper_indices] = np.random.normal(loc=0, scale=1, size=len(upper_indices[0]))
    
    return J

def Fully_connected_1D(system_size:int):
    """
    Create a fully connected interaction matrix in one dimension.

    This function returns a matrix of shape (system_size, system_size) 
    where the strictly upper triangular entries (excluding the diagonal) 
    are set to 1, indicating a fully connected network in one direction (i.e., 
    connections exist from each node to every other node with a higher index).
    The lower triangular entries (including the diagonal) remain zero.

    Parameters:
        system_size (int): The number of nodes in the system. 
                           Determines the dimensions of the output matrix.

    Returns:
        numpy.ndarray: A 2D NumPy array of shape (system_size, system_size) with the 
                       strictly upper triangular entries set to 1 and zeros elsewhere.
                       
    Example:
        >>> matrix = Fully_connected_1D(4)
        >>> print(matrix)
        [[0. 1. 1. 1.]
         [0. 0. 1. 1.]
         [0. 0. 0. 1.]
         [0. 0. 0. 0.]]
    """
    # Return the strictly upper triangular part of a matrix filled with ones.
    return np.triu(np.ones((system_size, system_size)), k=1)/system_size

def Nearest_neighbor_1D(system_size: int):
    """
    Create a nearest-neighbor interaction matrix for a 1D Ising chain 
    (open boundary conditions).

    This function returns a matrix of shape (system_size, system_size) 
    with ones on the first superdiagonal [i, i+1] and zeros elsewhere. 
    In 0-based indexing, that means positions (0,1), (1,2), ..., (N-2, N-1) 
    will be set to 1, while all other entries remain zero.
    
    Parameters:
        system_size (int): The number of spins (or sites) in the 1D chain.

    Returns:
        numpy.ndarray: A 2D NumPy array of shape (system_size, system_size) 
                       with 1s in the nearest-neighbor (i, i+1) positions.
                       
    Example:
        >>> matrix = Nearest_neighbor_1D(4)
        >>> print(matrix)
        [[0. 1. 0. 0.]
         [0. 0. 1. 0.]
         [0. 0. 0. 1.]
         [0. 0. 0. 0.]]
    """
    # np.diag takes a 1D array and puts it on the specified diagonal (k=1 is the superdiagonal).
    return np.diag(np.ones(system_size - 1), k=1)




