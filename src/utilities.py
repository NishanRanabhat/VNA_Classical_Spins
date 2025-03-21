import torch
import numpy as np
import random
from neural_network_ansatz import binary_disordered_RNNwavefunction, binary_disordered_RNNwavefunction_weight_sharing
from objective_function import One_dimensional_spin_model
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Dataset
import os
import json
from pathlib import Path
import time
wd = os.getcwd() 

def set_annealing_schedule(warmup_on, scheduler, warmup_time, annealing_time, equilibrium_time, T0, Tf, ftype):

  """
  Create an annealing schedule for the temperature parameter during training.

  The function constructs a temperature schedule based on the specified scheduler type 
  ("exponential", "linear", or "quadratic") and whether a warmup period is applied.
  The schedule is built using the total number of steps derived from annealing_time and 
  equilibrium_time (and warmup_time if applicable). The resulting temperature list is 
  returned as a torch tensor of the specified floating point type.

  Parameters:
    warmup_on (str): Indicates whether a warmup period is used ("True" or "False").
    scheduler (str): Type of scheduler to use ("exponential", "linear", or "quadratic").
    warmup_time (int): Number of steps dedicated to the warmup period.
    annealing_time (int): Number of annealing steps.
    equilibrium_time (int): Number of steps per annealing step.
    T0 (float): Initial temperature.
    Tf (float): Final temperature.
    ftype (torch.dtype): Floating point data type for the temperature tensor.

  Returns:
    torch.Tensor: A tensor containing the scheduled temperature values.
  """
  nsteps = annealing_time*equilibrium_time + 1
  num_steps = annealing_time*equilibrium_time + warmup_time + 1 

  Temperature_list = None

  if warmup_on == "False" and scheduler == "exponential":
    Temperature_list = np.ones(nsteps+1)*T0
    for i in range(nsteps):
      if i % equilibrium_time == 0:
        annealing_step = i/equilibrium_time
        Temperature_list[i] = T0*(Tf/T0)**(annealing_step/annealing_time) 
      Temperature_list[i+1] = Temperature_list[i]

  elif warmup_on == "False" and scheduler == "linear":
    Temperature_list = np.ones(nsteps+1)*T0
    for i in range(nsteps):
      if i % equilibrium_time == 0:
        annealing_step = i/equilibrium_time
        Temperature_list[i] = Tf - (Tf-T0)*(1.0-annealing_step/annealing_time) 
      Temperature_list[i+1] = Temperature_list[i]

  elif warmup_on == "False" and scheduler == "quadratic":
    Temperature_list = np.ones(nsteps+1)*T0
    for i in range(nsteps):
      if i % equilibrium_time == 0:
        annealing_step = i/equilibrium_time
        Temperature_list[i] = Tf - (Tf-T0)*(1.0-annealing_step/annealing_time)**2 
      Temperature_list[i+1] = Temperature_list[i]
  
  elif warmup_on == "True" and scheduler == "exponential":
    Temperature_list = np.ones(num_steps+1)*T0
    for i in range(num_steps):
      if i % equilibrium_time == 0 and i>=warmup_time:
        annealing_step = (i-warmup_time)/equilibrium_time
        Temperature_list[i] = T0*(Tf/T0)**(annealing_step/annealing_time)
      Temperature_list[i+1] = Temperature_list[i]

  elif warmup_on == "True" and scheduler == "linear":
    Temperature_list = np.ones(num_steps+1)*T0
    for i in range(num_steps):
      if i % equilibrium_time == 0 and i>=warmup_time:
        annealing_step = (i-warmup_time)/equilibrium_time
        Temperature_list[i] = Tf - (Tf-T0)*(1.0-annealing_step/annealing_time)
      Temperature_list[i+1] = Temperature_list[i]

  elif warmup_on == "True" and scheduler == "quadratic":
    Temperature_list = np.ones(num_steps+1)*T0
    for i in range(num_steps):
      if i % equilibrium_time == 0 and i>=warmup_time:
        annealing_step = (i-warmup_time)/equilibrium_time
        Temperature_list[i] = Tf - (Tf-T0)*(1.0-annealing_step/annealing_time)**2
      Temperature_list[i+1] = Temperature_list[i]

  Temperature_list = torch.tensor(Temperature_list, dtype=ftype)

  return Temperature_list


def model_ansatz(key:str,system_size:int,input_dim:int, num_layers:int,ftype:torch.dtype,**kwargs):

  """
  Initialize the variational ansatz model.

  This function creates an instance of the binary_disordered_RNNwavefunction using the
  provided parameters and any additional optional parameters specified in kwargs.

  Parameters:
    key (str): Specifies the type of RNN cell ("vanilla" or "gru").
    system_size (int): Number of spins (sites) in the system.
    input_dim (int): Dimensionality of the input (e.g., 2 for binary systems).
    num_layers (int): Number of recurrent layers per site.
    ftype (torch.dtype): Data type for model parameters.
    **kwargs: Additional optional parameters, including:
        - num_units (int): Number of units in each RNN cell (default: 10).
        - seed (int): Random seed for reproducibility (default: 111).
        - device: Device on which the model is allocated (e.g., 'cpu' or 'cuda').

  Returns:
    binary_disordered_RNNwavefunction: Instantiated ansatz model.
  """

  num_units = kwargs.get('num_units',10)
  weight_sharing=kwargs.get('weight_sharing',"true")
  device = kwargs.get('device')

  if weight_sharing == "True":
    ansatz = binary_disordered_RNNwavefunction_weight_sharing(key,system_size,input_dim,num_layers,activation="relu",units=num_units,type=ftype,device=device)
  else:
    ansatz =  binary_disordered_RNNwavefunction(key,system_size,input_dim,num_layers,activation="relu",units=num_units,type=ftype,device=device)

  return ansatz

def model_class(system_size,J_matrix,device,ftype):

  """
  Initialize the one-dimensional spin model.

  Parameters:
    system_size (int): Total number of spins in the system.
    J_matrix (array-like): Coupling matrix for the spin interactions.
    device: Device on which the model is allocated (e.g., 'cpu' or 'cuda').
    ftype (torch.dtype): Data type for model parameters.

  Returns:
    One_dimensional_spin_model: Instantiated spin model.
  """

  model = One_dimensional_spin_model(system_size,J_matrix,device,ftype)

  return model

def optimizer_init(ansatz,learningrate = 1e-3,optimizer_type="adam"):

  """
  Initialize the optimizer for the ansatz model.

  Parameters:
    ansatz (torch.nn.Module): The model parameters to be optimized.
    learningrate (float, optional): Learning rate for the optimizer (default: 1e-3).
    optimizer_type (str, optional): Type of optimizer ("adam", "rmsprop", or "sgd").

  Returns:
    torch.optim.Optimizer: The instantiated optimizer.
  """

  if optimizer_type == "adam":
    optimizer = torch.optim.Adam(ansatz.parameters(), lr= learningrate) 
  elif optimizer_type == "rmsprop":
    optimizer = torch.optim.RMSprop(ansatz.parameters(), lr= learningrate, weight_decay=0.005525040440839694)
  elif optimizer_type == "sgd":
    optimizer = torch.optim.SGD(ansatz.parameters(), lr= learningrate)

  return optimizer

def scheduler_init(optimizer, num_steps:int, scheduler_name="None"):

  """
  Initialize the learning rate scheduler.

  Parameters:
    optimizer (torch.optim.Optimizer): The optimizer whose learning rate will be scheduled.
    num_steps (int): Number of steps for scheduling, used by some schedulers.
    scheduler_name (str, optional): Type of scheduler to use. Options include:
        "StepLR", "ReduceLROnPlateau", "MultiStep", "CosineAnnealingLR", "MultiplicativeLR", "Exponential", or "None".

  Returns:
    Scheduler: The instantiated learning rate scheduler, or None if scheduler_name is "None".
  """
  
  decay_factor = 0.9999

  if scheduler_name == "StepLR":
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=num_steps, gamma=0.1)
  elif scheduler_name == "ReduceLROnPlateau":
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
  elif scheduler_name == "MultiStep":
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2500,2800], gamma=0.1)
  elif scheduler_name == "CosineAnnealingLR":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
  elif scheduler_name == "MultiplicativeLR":
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: 0.9)
  elif scheduler_name == "Exponential":
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_factor)
  elif scheduler_name == "None":
    pass

  return scheduler

def seed_everything(seed, rank):

  """
  Seed all random number generators for reproducibility across processes.

  This function sets the seed for Python's random, NumPy, and PyTorch (both CPU and GPU) 
  random number generators. A unique seed is calculated for each GPU process by adding the rank.

  Parameters:
    seed (int): Base seed value.
    rank (int): Rank of the current process, used to ensure unique seeding across GPUs.
  """

  seed = seed + rank #unique seed because of rank

  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

#def input_data(num_samples:int,input_size:int,ftype:torch.dtype):

  """
  Create dummy input data for testing or initialization.

  Generates a dummy input tensor of shape (num_samples, input_size) with all zeros,
  except for the first column which is set to ones.

  Parameters:
    num_samples (int): Number of samples (rows) in the dummy input.
    input_size (int): Number of features (columns) per sample.
    ftype (torch.dtype): Data type for the tensor.

  Returns:
    torch.Tensor: The dummy input tensor.
  """

#  dummy_input = torch.zeros(num_samples,input_size, dtype=ftype)
#  dummy_input[:,0] = torch.ones(num_samples, dtype=ftype)

#  return dummy_input

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

def prepare_dataloader(dataset, world_size: int, rank, batch_size: int):

  """
  Prepare a distributed DataLoader for use in multi-GPU training.

  Wraps the provided dataset with a DistributedSampler to ensure each process 
  gets a unique subset of the data, and returns a DataLoader with the specified 
  batch size and worker configuration.

  Parameters:
    dataset (Dataset): The dataset to load.
    world_size (int): Total number of processes (GPUs) used in training.
    rank (int): Rank of the current process.
    batch_size (int): Number of samples per batch.

  Returns:
    DataLoader: Configured DataLoader for distributed training.
  """
    
  return DataLoader(
    dataset,
    batch_size=batch_size,
    pin_memory=True,
    shuffle=False,
    sampler=DistributedSampler(dataset, num_replicas=world_size, rank=rank),
    num_workers=4,
    persistent_workers=True)
