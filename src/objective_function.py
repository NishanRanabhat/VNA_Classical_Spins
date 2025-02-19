import numpy as np
import torch

class One_dimensional_spin_model:
    def __init__(self, system_size,J_matrix,device=torch.device("cpu"), dtype=torch.float32):
      """
      Initialize the Sherrington-Kirkpatrick model.
        
      Parameters:
        - N: Number of spins.
        - J_matrix: The interaction matrix, upper triangular with zeros on diagonal
        - device: Torch device.
        - dtype: Torch data type.
      """

      self.N = system_size
      self.J_matrix = J_matrix
      self.device = device
      self.type = dtype

      #convert the J_matrix into a torch tensor and put it in the current device with the correct type
      self.J_matrix = torch.tensor(self.J_matrix).to(dtype=self.type, device=self.device)

      """
      encod to convert the 0 and 1 to -1.0 and 1.0 
      """
      encod = torch.zeros(2, dtype=self.type).to(device) 
      encod[0] = -1
      encod[1] = 1
      self.get_encoding = encod

    def get_configs(self,x):
      """
      converts the 0/1 samples x into -1/1 configs 
        
      Parameters:
        - x: A torch tensor of shape (num_samples,N) with entries 0 or 1.
        
      Returns:
        - A torch tensor of shape (num_samples,N) with entries -1 or 1.
      """

      x_one_hot = torch.nn.functional.one_hot(x.long(), num_classes=2).type(self.type)
      return torch.sum(torch.multiply(self.get_encoding, x_one_hot), dim=2) 

    def energy(self, configs):
        """
        Compute the energy of a spin configuration.
        
        Parameters:
          - config: A torch tensor of shape (num_samples,N) with entries +1 or -1.
        
        Returns:
          - Energy (scalar).
        """
        # Calculate energy: E = - sum_{i<j} J_{ij} S_i S_j.
        # We can write it in a symmetric form:
        energy = -1.0*torch.sum(configs.t() * (self.J_matrix @ configs.t()),dim=0)
        return energy