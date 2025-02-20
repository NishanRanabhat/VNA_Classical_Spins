import torch
import numpy as np
import random
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import json
from pathlib import Path
import time
import torch.profiler
from torch.profiler import schedule
import os
import json
from pathlib import Path
import time
wd = os.getcwd() 


class VNA_trainer:

  """
  Trainer for Variational Neural Ansatz (VNA) models in a distributed training setup.

  This class wraps a variational ansatz neural network (the "ansatz") with training utilities,
  enabling distributed training with PyTorch's DistributedDataParallel (DDP). It performs
  forward passes, computes energies from a spin model, calculates the cost (loss), and
  updates the network parameters using an optimizer and a learning rate scheduler.
    
  Attributes:
    gpu_id (int): Identifier for the current GPU.
    ansatz (DDP): DistributedDataParallel wrapped neural network model.
    train_data (DataLoader): DataLoader providing the training data batches.
    optimizer (torch.optim.Optimizer): Optimizer used for training.
    scheduler: Learning rate scheduler to adjust the optimizer's learning rate.
    model: Spin model that provides energy and configuration conversions.
  """

  def __init__(self,ansatz: torch.nn.Module,train_data: DataLoader,optimizer: torch.optim.Optimizer,scheduler,model,gpu_id: int):
    
    self.gpu_id = gpu_id
    self.ansatz = ansatz.to(gpu_id)
    self.ansatz = DDP(ansatz, device_ids=[gpu_id])
    self.train_data = train_data
    self.optimizer = optimizer
    self.scheduler = scheduler
    self.model = model

  def _run_batch(self, source,Temperature):
    
    """
    Process a single batch of training data.

    Performs a forward pass through the ansatz model, obtains a spin configuration from the
    variational neural ansatz, computes the local energy, calculates the cost function, and 
    performs a backpropagation step to update the model's parameters.

    Parameters:
      source (torch.Tensor): Input features for the batch.
      Temperature (float): Temperature parameter used to weight the log-probabilities.

    Returns:
      numpy.ndarray: The detached local energies computed for the batch, converted to a NumPy array.
    """

    _ = self.ansatz(source)

    configs = self.model.get_configs(self.ansatz.module.samples)
    local_energies = self.model.energy(configs)
    magnetization = self.model.magnetization(configs)
    log_probs = self.ansatz.module.log_probs
    Floc = local_energies + Temperature * log_probs
    cost = torch.mean(log_probs * Floc.detach()) - torch.mean(log_probs) * torch.mean(Floc.detach())
      
    self.optimizer.zero_grad() 
    cost.backward()
    self.optimizer.step()

    return Floc.detach().cpu().numpy(), magnetization.detach().cpu().numpy()
    
  def _run_epoch(self, epoch:int,Temperature):
    
    """
    Run a complete training epoch.

    Iterates over all batches provided by the DataLoader, processing each batch and
    collecting the corresponding local energies.

    Parameters:
      epoch (int): The current epoch number.
      Temperature (float): Temperature parameter used in cost computation.

    Returns:
      numpy.ndarray: Concatenated local energies from all batches in the epoch.
    """

    self.train_data.sampler.set_epoch(epoch)
    epoch_Floc, epoch_mag = [],[]

    for batch_idx, source in enumerate(self.train_data):
      source = source.to(self.gpu_id)   
      Floc, mag = self._run_batch(source,Temperature)
      epoch_Floc.append(Floc)
      epoch_mag.append(mag)

    return np.concatenate(epoch_Floc),np.concatenate(epoch_mag)

  def _gather(self, data):

    """
    Gather data across all GPUs participating in the distributed training.

    Uses torch.distributed.all_gather to collect data (either tensors or convertible data)
    from all processes and returns a flattened NumPy array containing the gathered data.

    Parameters:
      data: Data collected from the current GPU. Can be a torch.Tensor or convertible to one.

    Returns:
      numpy.ndarray: A flattened array of the gathered data from all GPUs.
    """
    
    if isinstance(data, torch.Tensor) and data.device == torch.device(f'cuda:{self.gpu_id}'):
      tensor = data
    else:
      # If it's not a tensor or not on the correct GPU, convert it
      tensor = torch.tensor(data, dtype=torch.float32).to(self.gpu_id)

    gathered_tensor = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]#create a empty list to keep all gathered output

    dist.all_gather(gathered_tensor, tensor)#gather outputs from all GPUs

    return np.concatenate([t.cpu().numpy() for t in gathered_tensor])#convert the gathered outputs into numpy and return the flattend version

  def train(self, total_epochs: int,Temperature_list,gather_interval:int):

    """
    Train the ansatz model for a specified number of epochs.

    Iterates over training epochs, updating the model and adjusting the learning rate. 
    At specified intervals, gathers the local energy data across GPUs, computes the mean energy,
    and prints the statistics.

    Parameters:
      total_epochs (int): Total number of training epochs.
      Temperature_list (list or array-like): A list of temperature values for each epoch.
      gather_interval (int): Interval (in epochs) at which to gather and report statistics.

    Returns:
      numpy.ndarray: An array containing the mean local energy collected at each gathering interval.
    """

    all_Floc, all_mag = [],[]

    for epoch in range(total_epochs):

      Temperature = Temperature_list[epoch]
      Floc,mag = self._run_epoch(epoch,Temperature)
      self.scheduler.step()

      if epoch % gather_interval == 0 or epoch == 0 or epoch == total_epochs-1:

        gathered_Floc = self._gather(Floc)
        gathered_mag = self._gather(mag)
        all_Floc.append(np.mean(gathered_Floc))
        all_mag.append(np.mean(gathered_mag))
 

      if self.gpu_id == 0:
        print("Energy=",np.mean(gathered_Floc),np.var(gathered_Floc))
        print("magnetization=",np.mean(gathered_mag))

    return np.array(all_Floc), np.array(all_mag)