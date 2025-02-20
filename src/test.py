import torch
import numpy as np

rows, cols = 3, 4
low, high = 0, 10

# Initialize the 2D tensor with random integers
tensor = torch.randint(low=low, high=high, size=(rows, cols))
print(tensor)

col_sum = tensor.sum(dim=1)
print(col_sum)