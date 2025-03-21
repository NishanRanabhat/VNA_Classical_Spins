import torch
import numpy as np

def input_data(num_samples: int, input_size: int, ftype: torch.dtype):
    """
    Create dummy input data for testing or initialization.

    Generates a dummy input tensor of shape (num_samples, input_size) with each row
    randomly starting with either an "all up" or "all down" indicator in the first column.
    Other features are set to zeros.

    Parameters:
        num_samples (int): Number of samples (rows) in the dummy input.
        input_size (int): Number of features (columns) per sample.
        ftype (torch.dtype): Data type for the tensor.

    Returns:
        torch.Tensor: The dummy input tensor.
    """
    dummy_input = torch.zeros(num_samples, input_size, dtype=ftype)
    # Randomly choose for each sample whether the first column is 1 (up) or 0 (down)
    dummy_input[:, 0] = torch.randint(0, 2, (num_samples,), dtype=torch.int64).to(ftype)
    return dummy_input

print(input_data(10, 2,int))
