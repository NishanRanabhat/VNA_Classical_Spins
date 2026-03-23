import torch
import numpy as np
import random
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
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

  def __init__(self, ansatz: torch.nn.Module,train_data: DataLoader,train_batch_size:int, 
               optimizer: torch.optim.Optimizer, interaction_name:str, weight_sharing:str, 
               scheduler , model, gpu_id: int):
    
    self.gpu_id = gpu_id
    self.nlayers = ansatz.n_layers
    self.key = ansatz.key
    self.system_size = ansatz.N
    self.hiden_dim = ansatz.hidden_dim
    self.ansatz = ansatz.to(gpu_id)
    self.ansatz = DDP(ansatz, device_ids=[gpu_id])
    self.train_data = train_data
    self.optimizer = optimizer
    self.scheduler = scheduler
    self.model = model
    self.train_batch_size = train_batch_size
    self.interaction_name = interaction_name
    self.weight_sharing = weight_sharing

  def _run_batch(self, source, Temperature):
    
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
    Eloc = self.model.energy(configs)
    magnetization = self.model.magnetization(configs)
    log_probs = self.ansatz.module.log_probs
    Floc = Eloc + Temperature * log_probs
    cost = torch.mean(log_probs * Floc.detach()) - torch.mean(log_probs) * torch.mean(Floc.detach())

    self.optimizer.zero_grad() 
    cost.backward()
    self.optimizer.step()
    return Eloc.detach().cpu().numpy(), magnetization.detach().cpu().numpy(), log_probs.detach().cpu().numpy(), Floc.detach().cpu().numpy()
    
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
    epoch_Eloc,epoch_mag,epoch_mag2,epoch_mag4,epoch_log_probs,epoch_Floc = [],[],[],[],[],[]

    for batch_idx, source in enumerate(self.train_data):
      source = source.to(self.gpu_id)   
      Eloc,mag,log_probs,Floc = self._run_batch(source,Temperature)
      epoch_Eloc.append(Eloc)
      epoch_mag.append(mag)
      epoch_mag2.append(mag**2)
      epoch_mag4.append(mag**4)
      epoch_log_probs.append(log_probs)
      epoch_Floc.append(Floc)
      
    return np.concatenate(epoch_Eloc),np.concatenate(epoch_mag),np.concatenate(epoch_mag2),np.concatenate(epoch_mag4),np.concatenate(epoch_log_probs),np.concatenate(epoch_Floc)

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

  def train(self, annealing_time, total_epochs:int, Temperature_list, gather_interval:int):

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

    Temps, all_Eloc, all_varEloc, all_varEloc, all_mag, all_varmag, all_mag2, all_varmag2, all_mag4, all_varmag4, all_log_probs,all_Floc, all_VarFloc = [],[],[],[],[],[],[],[],[],[],[],[],[]

    for epoch in range(total_epochs):

      Temperature = Temperature_list[epoch]
      Eloc,mag,mag2,mag4,log_probs,Floc = self._run_epoch(epoch,Temperature)
      self.scheduler.step()

      if epoch % gather_interval == 0 or epoch == 0 or epoch == total_epochs-1:

        gathered_Eloc = self._gather(Eloc)
        gathered_mag = self._gather(mag)
        gathered_mag2 = self._gather(mag2)
        gathered_mag4 = self._gather(mag4)
        gathered_log_probs = self._gather(log_probs)
        gathered_Floc = self._gather(Floc)
        Temps.append(Temperature)
        all_Eloc.append(np.mean(gathered_Eloc))
        all_varEloc.append(np.var(gathered_Eloc))
        all_mag.append(np.mean(gathered_mag))
        all_varmag.append(np.var(gathered_mag))
        all_mag2.append(np.mean(gathered_mag2))
        all_varmag2.append(np.var(gathered_mag2))
        all_mag4.append(np.mean(gathered_mag4))
        all_varmag4.append(np.var(gathered_mag4))
        all_log_probs.append(np.mean(gathered_log_probs))
        all_Floc.append(np.mean(gathered_Floc))
        all_VarFloc.append(np.var(gathered_Floc))

      if self.gpu_id == 0:
        print("Ti={:.4f}, Eloc={:.7f}, variance={:.7f}".format(Temperature,np.mean(gathered_Eloc),np.var(gathered_Eloc)))
    savepath = f'../output_files/{self.interaction_name}_sample{self.train_batch_size}_layer{self.nlayers}_Nh{self.hiden_dim}_ws_{self.weight_sharing}_tau{annealing_time}/'
    os.makedirs(savepath, exist_ok=True)

    if self.gpu_id == 0:
      np.save(f'{savepath}VNAtrain_TemperatureList={Temperature}_N={self.system_size}.npy', np.array(Temps))
      np.save(f'{savepath}VNAtrain_Eloc_temperature={Temperature}_N={self.system_size}.npy', np.array(all_Eloc))
      np.save(f'{savepath}VNAtrain_varEloc_temperature={Temperature}_N={self.system_size}.npy', np.array(all_varEloc))
      np.save(f'{savepath}VNAtrain_mag_temperature={Temperature}_N={self.system_size}.npy', np.array(all_mag))
      np.save(f'{savepath}VNAtrain_varmag_temperature={Temperature}_N={self.system_size}.npy', np.array(all_varmag))
      np.save(f'{savepath}VNAtrain_mag2_temperature={Temperature}_N={self.system_size}.npy', np.array(all_mag2))
      np.save(f'{savepath}VNAtrain_varmag2_temperature={Temperature}_N={self.system_size}.npy', np.array(all_varmag2))
      np.save(f'{savepath}VNAtrain_mag4_temperature={Temperature}_N={self.system_size}.npy', np.array(all_mag4))
      np.save(f'{savepath}VNAtrain_varmag4_temperature={Temperature}_N={self.system_size}.npy', np.array(all_varmag4))
      np.save(f'{savepath}VNAtrain_log_probs_temperature={Temperature}_N={self.system_size}.npy', np.array(all_log_probs))
      np.save(f'{savepath}VNAtrain_Floc_temperature={Temperature}_N={ self.system_size}.npy', np.array(all_Floc))
      np.save(f'{savepath}VNAtrain_varFloc_temperature={Temperature}_N={ self.system_size}.npy', np.array(all_VarFloc))

    return np.array(all_Floc), np.array(all_mag)

class Brute_Gradient_Descent:

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

  def __init__(self,ansatz: torch.nn.Module,train_data: DataLoader,train_batch_size: int, 
              optimizer: torch.optim.Optimizer,interaction_name: str,weight_sharing: str,
              scheduler,model,gpu_id: int):
    
    self.gpu_id = gpu_id
    self.nlayers = ansatz.n_layers
    self.key = ansatz.key
    self.system_size = ansatz.N
    self.hiden_dim = ansatz.hidden_dim
    self.ansatz = ansatz.to(gpu_id)
    self.ansatz = DDP(ansatz, device_ids=[gpu_id])
    self.train_data = train_data
    self.optimizer = optimizer
    self.scheduler = scheduler
    self.model = model
    self.train_batch_size = train_batch_size
    self.interaction_name = interaction_name
    self.weight_sharing = weight_sharing

  def count_neg_and_pos(self, x: torch.Tensor) -> torch.Tensor:
    """
    Given a tensor x of shape [N, M] containing only -1 and 1,
    returns a tensor of shape [N, 2], where for each row i:
      output[i, 0] = number of -1 in row i
      output[i, 1] = number of  1 in row i
    """
    count_neg = (x == -1).sum(dim=1)
    count_pos = (x == 1).sum(dim=1)
    return torch.stack([count_neg, count_pos], dim=1)

  def _run_batch(self, source, Temperature):
    
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
    Eloc = self.model.energy(configs)
    magnetization = self.model.magnetization(configs)
    log_probs = self.ansatz.module.log_probs
    Floc = Eloc + Temperature * log_probs
    cost = torch.mean(log_probs * Floc.detach()) - torch.mean(log_probs) * torch.mean(Floc.detach())

    self.optimizer.zero_grad() 
    cost.backward()
    self.optimizer.step()

    return Eloc.detach().cpu().numpy(),magnetization.detach().cpu().numpy(),log_probs.detach().cpu().numpy(),Floc.detach().cpu().numpy()
    
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
    epoch_Eloc,epoch_mag,epoch_mag2,epoch_mag4,epoch_log_probs,epoch_Floc = [],[],[],[],[],[]

    for batch_idx, source in enumerate(self.train_data):
      source = source.to(self.gpu_id)   
      Eloc,mag,log_probs,Floc = self._run_batch(source,Temperature)
      epoch_Eloc.append(Eloc)
      epoch_mag.append(mag)
      epoch_mag2.append(mag**2)
      epoch_mag4.append(mag**4)
      epoch_log_probs.append(log_probs)
      epoch_Floc.append(Floc)      
      
    return np.concatenate(epoch_Eloc),np.concatenate(epoch_mag),np.concatenate(epoch_mag2),np.concatenate(epoch_mag4),np.concatenate(epoch_log_probs),np.concatenate(epoch_Floc)

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

  def train(self, annealing_time, total_epochs: int,Temperature, gather_interval:int):

    """
    Train the ansatz model for a specified number of epochs.

    Iterates over training epochs, updating the model and adjusting the learning rate. 
    At specified intervals, gathers the local energy data across GPUs, computes the mean energy,
    and prints the statistics.

    Parameters:
      total_epochs (int): Total number of training epochs.
      Temperature: Temperature
      gather_interval (int): Interval (in epochs) at which to gather and report statistics.

    Returns:
      numpy.ndarray: An array containing the mean local energy collected at each gathering interval.
    """

    all_Eloc, all_varEloc, all_mag, all_varmag, all_mag2, all_varmag2, all_mag4, all_varmag4, all_log_probs,all_Floc, all_VarFloc = [],[],[],[],[],[],[],[],[],[],[]

    for epoch in range(total_epochs):

      Eloc,mag,mag2,mag4,log_probs,Floc = self._run_epoch(epoch,Temperature)
      self.scheduler.step()

      if epoch % gather_interval == 0 or epoch == 0 or epoch == total_epochs-1:

        gathered_Eloc = self._gather(Eloc)
        gathered_mag = self._gather(mag)
        gathered_mag2 = self._gather(mag2)
        gathered_mag4 = self._gather(mag4)
        gathered_log_probs = self._gather(log_probs)
        gathered_Floc = self._gather(Floc)
        all_Eloc.append(np.mean(gathered_Eloc))
        all_varEloc.append(np.var(gathered_Eloc))
        all_mag.append(np.mean(gathered_mag))
        all_varmag.append(np.var(gathered_mag))
        all_mag2.append(np.mean(gathered_mag2))
        all_varmag2.append(np.var(gathered_mag2))
        all_mag4.append(np.mean(gathered_mag4))
        all_varmag4.append(np.var(gathered_mag4))
        all_log_probs.append(np.mean(gathered_log_probs))
        all_Floc.append(np.mean(gathered_Floc))
        all_VarFloc.append(np.var(gathered_Floc))
 

      if self.gpu_id == 0 and epoch % 1 == 0:
        print("Epoch {}/{}: Eloc={:.4f}, variance={:.4f}".format(epoch+1, total_epochs, np.mean(gathered_Eloc), np.var(gathered_Eloc)))

    # Create output directory if it doesn't exist
    savepath = f'../output_files/{self.interaction_name}_sample{self.train_batch_size}_layer{self.nlayers}_Nh{self.hiden_dim}_ws_{self.weight_sharing}_tau{annealing_time}/'
    os.makedirs(savepath, exist_ok=True)

    if self.gpu_id == 0:
      np.save(f'{savepath}BGD_Eloc_temperature={Temperature}_N={self.system_size}.npy', np.array(all_Eloc))
      np.save(f'{savepath}BGD_varEloc_temperature={Temperature}_N={self.system_size}.npy', np.array(all_varEloc))
      np.save(f'{savepath}BGD_mag_temperature={Temperature}_N={self.system_size}.npy', np.array(all_mag))
      np.save(f'{savepath}BGD_varmag_temperature={Temperature}_N={self.system_size}.npy', np.array(all_varmag))
      np.save(f'{savepath}BGD_mag2_temperature={Temperature}_N={self.system_size}.npy', np.array(all_mag2))
      np.save(f'{savepath}BGD_varmag2_temperature={Temperature}_N={self.system_size}.npy', np.array(all_varmag2))
      np.save(f'{savepath}BGD_mag4_temperature={Temperature}_N={self.system_size}.npy', np.array(all_mag4))
      np.save(f'{savepath}BGD_varmag4_temperature={Temperature}_N={self.system_size}.npy', np.array(all_varmag4))        
      np.save(f'{savepath}BGD_log_probs_temperature={Temperature}_N={self.system_size}.npy', np.array(all_log_probs))
      np.save(f'{savepath}BGD_Floc_temperature={Temperature}_N={self.system_size}.npy', np.array(all_Floc))
      np.save(f'{savepath}BGD_varFloc_temperature={Temperature}_N={self.system_size}.npy', np.array(all_VarFloc))


    return np.array(all_Floc), np.array(all_mag)

class VQA_trainer:

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

  def __init__(self,ansatz: torch.nn.Module,train_data: DataLoader,train_batch_size:int, 
              optimizer: torch.optim.Optimizer, interaction_name: str,weight_sharing: str,
              scheduler,model,gpu_id: int):
    
    self.gpu_id = gpu_id
    self.nlayers = ansatz.n_layers
    self.key = ansatz.key
    self.system_size = ansatz.N
    self.hiden_dim = ansatz.hidden_dim
    self.ansatz = ansatz.to(gpu_id)
    self.ansatz = DDP(ansatz, device_ids=[gpu_id])
    self.train_data = train_data
    self.optimizer = optimizer
    self.scheduler = scheduler
    self.model = model
    self.train_batch_size = train_batch_size
    self.interaction_name = interaction_name
    self.weight_sharing = weight_sharing

  def generate_spin_flips(self, samples):
    batch_size, system_size = samples.shape
    flipped_samples = samples.unsqueeze(0).expand(system_size, -1, -1).clone()
    idx = torch.arange(start=system_size-1, end=-1, step=-1, device=samples.device)
    flipped_samples[torch.arange(system_size), :, idx] *= -1
    flipped_samples[torch.arange(system_size), :, idx] += 1
    
    return torch.cat((flipped_samples, samples.unsqueeze(0)), dim=0)
  
  def compute_wavefunction_ratio(self, ansatz, samples):
    batch_size, system_size = samples.shape
    flipped_spins = self.generate_spin_flips(samples)
    log_probs = ansatz.log_probability(0.5*(flipped_spins+1.).reshape((system_size+1)*batch_size, system_size)).reshape(-1, batch_size)
    log_wf_ratios = 0.5*(log_probs[:system_size,:]-log_probs[-1,:].unsqueeze(0))
    return torch.exp(log_wf_ratios)

  def _run_batch(self, source, Temperature):
    
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
    Epot = self.model.energy(configs)
    magnetization = self.model.magnetization(configs)
    log_probs = self.ansatz.module.log_probs
    with torch.no_grad():
      wave_function_ratios = self.compute_wavefunction_ratio(self.ansatz.module, self.ansatz.module.samples)
    Eloc = Epot - Temperature * torch.sum(wave_function_ratios, dim=0)
    cost = torch.mean(log_probs * Eloc.detach()) - torch.mean(log_probs) * torch.mean(Eloc.detach())

    self.optimizer.zero_grad() 
    cost.backward()
    self.optimizer.step()
    return Eloc.detach().cpu().numpy(), torch.abs(magnetization).detach().cpu().numpy(), log_probs.detach().cpu().numpy()
    
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
    epoch_Eloc,epoch_mag,epoch_mag2,epoch_mag4,epoch_log_probs = [],[],[],[],[]

    for batch_idx, source in enumerate(self.train_data):
      source = source.to(self.gpu_id)   
      Eloc,mag,log_probs = self._run_batch(source,Temperature)
      epoch_Eloc.append(Eloc)
      epoch_mag.append(mag)
      epoch_mag2.append(mag**2)
      epoch_mag4.append(mag**4)
      epoch_log_probs.append(log_probs)
      
    return np.concatenate(epoch_Eloc),np.concatenate(epoch_mag),np.concatenate(epoch_mag2),np.concatenate(epoch_mag4),np.concatenate(epoch_log_probs)

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

  def train(self, annealing_time, total_epochs: int,Temperature_list,gather_interval:int):

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
    # Magnetization mismatch comes from the training process

    Temps, all_Eloc, all_varEloc, all_varEloc, all_mag, all_varmag, all_mag2, all_varmag2, all_mag4, all_varmag4, all_log_probs = [],[],[],[],[],[],[],[],[],[],[]

    for epoch in range(total_epochs):

      Temperature = Temperature_list[epoch]
      Eloc,mag,mag2,mag4,log_probs = self._run_epoch(epoch,Temperature)
      self.scheduler.step()

      if epoch % gather_interval == 0 or epoch == 0 or epoch == total_epochs-1:

        gathered_Eloc = self._gather(Eloc)
        gathered_mag = self._gather(mag)
        gathered_mag2 = self._gather(mag2)
        gathered_mag4 = self._gather(mag4)
        gathered_log_probs = self._gather(log_probs)
        Temps.append(Temperature)
        all_Eloc.append(np.mean(gathered_Eloc))
        all_varEloc.append(np.var(gathered_Eloc))
        all_mag.append(np.mean(gathered_mag))
        all_varmag.append(np.var(gathered_mag))
        all_mag2.append(np.mean(gathered_mag2))
        all_varmag2.append(np.var(gathered_mag2))
        all_mag4.append(np.mean(gathered_mag4))
        all_varmag4.append(np.var(gathered_mag4))
        all_log_probs.append(np.mean(gathered_log_probs))

      if self.gpu_id == 0:
        print("Gamma={:.4f}, Eloc={:.7f}, variance={:.7f}".format(Temperature,np.mean(gathered_Eloc),np.var(gathered_Eloc)))

    savepath = f'../output_files/{self.interaction_name}_sample{self.train_batch_size}_layer{self.nlayers}_Nh{self.hiden_dim}_ws_{self.weight_sharing}_tau{annealing_time}/'
    os.makedirs(savepath, exist_ok=True)

    if self.gpu_id == 0:
      np.save(f'{savepath}VQAtrain_TemperatureList={Temperature}_N={self.system_size}.npy', np.array(Temps))
      np.save(f'{savepath}VQAtrain_Eloc_temperature={Temperature}_N={self.system_size}.npy', np.array(all_Eloc))
      np.save(f'{savepath}VQAtrain_varEloc_temperature={Temperature}_N={self.system_size}.npy', np.array(all_varEloc))
      np.save(f'{savepath}VQAtrain_mag_temperature={Temperature}_N={self.system_size}.npy', np.array(all_mag))
      np.save(f'{savepath}VQAtrain_varmag_temperature={Temperature}_N={self.system_size}.npy', np.array(all_varmag))
      np.save(f'{savepath}VQAtrain_mag2_temperature={Temperature}_N={self.system_size}.npy', np.array(all_mag2))
      np.save(f'{savepath}VQAtrain_varmag2_temperature={Temperature}_N={self.system_size}.npy', np.array(all_varmag2))
      np.save(f'{savepath}VQAtrain_mag4_temperature={Temperature}_N={self.system_size}.npy', np.array(all_mag4))
      np.save(f'{savepath}VQAtrain_varmag4_temperature={Temperature}_N={self.system_size}.npy', np.array(all_varmag4))
      np.save(f'{savepath}VQAtrain_log_probs_temperature={Temperature}_N={self.system_size}.npy', np.array(all_log_probs))

    return np.array(all_Eloc), np.array(all_mag)