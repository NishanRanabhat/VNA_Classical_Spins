import torch
import numpy as np
import random
from neural_network_ansatz import binary_disordered_RNNwavefunction
from objective_function import One_dimensional_spin_model
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Dataset
import os
import json
from pathlib import Path
import time
wd = os.getcwd() 

def set_annealing_schedule(warmup_on, scheduler, warmup_time, annealing_time, equilibrium_time, T0, Tf, ftype):
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

  # Extract optional parameters from kwargs with default
  num_units = kwargs.get('num_units',10)
  seed = kwargs.get('seed', 111)
  device = kwargs.get('device')

  ansatz =  binary_disordered_RNNwavefunction(key,system_size,input_dim,num_layers,activation="relu",units=num_units,seed=seed,type=ftype,device=device)

  return ansatz

def model_class(system_size,J_matrix,device,ftype):

    model = One_dimensional_spin_model(system_size,J_matrix,device,ftype)

    return model

def optimizer_init(ansatz,learningrate = 1e-3,optimizer_type="adam"):

  if optimizer_type == "adam":
    optimizer = torch.optim.Adam(ansatz.parameters(), lr= learningrate) 
  elif optimizer_type == "rmsprop":
    optimizer = torch.optim.RMSprop(ansatz.parameters(), lr= learningrate, weight_decay=0.005525040440839694)
  elif optimizer_type == "sgd":
    optimizer = torch.optim.SGD(ansatz.parameters(), lr= learningrate)

  return optimizer

def scheduler_init(optimizer, num_steps:int, scheduler_name="None"):
  
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

"""
This function takes seed as an argument and initializes all random number 
generators (random, numpy, torch, and torch.cuda) based on this seed. 
Each GPU process will receive a unique process_seed (calculated as seed + rank) 
ensuring different seeds across processes.
"""
def seed_everything(seed, rank):

    seed = seed + rank #unique seed because of rank

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def input_data(num_samples:int,input_size:int,ftype:torch.dtype):

  dummy_input = torch.zeros(num_samples,input_size, dtype=ftype)
  dummy_input[:,0] = torch.ones(num_samples, dtype=ftype)

  return dummy_input

def prepare_dataloader(dataset, world_size: int, rank, batch_size: int):
    
  return DataLoader(dataset, batch_size=batch_size, pin_memory=True, shuffle=False, sampler=DistributedSampler(dataset,num_replicas=world_size, rank=rank),num_workers = 4,persistent_workers=True)

