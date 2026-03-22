import numpy as np 

class ScalingDataset:
    """
    Simple container for finite-size scaling data.
    """
    def __init__(self, system_size,
                 x: np.ndarray,
                 y: np.ndarray,
                 err: np.ndarray):
        """
        Parameters:
            system_size: characteristic size L
            x: array of independent variable values (e.g., annealing times)
            y: array of dependent variable values (e.g., relative energies)
            err: array of uncertainties for y
        """
        self.L = float(system_size)
        self.x = np.array(x, dtype=float)
        self.y = np.array(y, dtype=float)
        self.err = np.array(err, dtype=float)