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
    source : input features
    """

    _ = self.ansatz(source)

    configs = self.model.get_configs(self.ansatz.module.samples)
    local_energies = self.model.energy(configs)
    log_probs = self.ansatz.module.log_probs
    Floc = local_energies + Temperature * log_probs
    cost = torch.mean(log_probs * Floc.detach()) - torch.mean(log_probs) * torch.mean(Floc.detach())
      
    self.optimizer.zero_grad() 
    cost.backward()
    self.optimizer.step()

    return Floc.detach().cpu().numpy()
    
  def _run_epoch(self, epoch:int,Temperature):

    """
    epoch : current epoch during training 
    """

    self.train_data.sampler.set_epoch(epoch)
    epoch_Floc = []

    for batch_idx, source in enumerate(self.train_data):
      source = source.to(self.gpu_id)   
      Floc = self._run_batch(source,Temperature)
      epoch_Floc.append(Floc)

    return np.concatenate(epoch_Floc)

  def _gather(self, data):

    """
    data: data colected from each GPU
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
    total_epochs : total training steps 
    """

    all_Floc = []

    for epoch in range(total_epochs):

      Temperature = Temperature_list[epoch]
      Floc = self._run_epoch(epoch,Temperature)
      self.scheduler.step()

      if epoch % gather_interval == 0 or epoch == 0 or epoch == total_epochs-1:

        gathered_Floc = self._gather(Floc)
        all_Floc.append(np.mean(gathered_Floc))
 

      if self.gpu_id == 0:
        print("Energy=",np.mean(gathered_Floc),np.var(gathered_Floc))

    return np.array(all_Floc)