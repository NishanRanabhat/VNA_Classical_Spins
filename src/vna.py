import torch
import numpy as np
import random
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from utilities import set_annealing_schedule, model_ansatz, model_class, optimizer_init, scheduler_init, seed_everything, input_data, prepare_dataloader
from trainer import VNA_trainer,Brute_Gradient_Descent
import os
import json
from pathlib import Path
import time
import os
import json
from pathlib import Path
import time
wd = os.getcwd() 

def setup(rank, world_size):

    """
    Set up the distributed process group for multi-GPU training.

    This function sets necessary environment variables and initializes the distributed
    process group using NCCL as the backend. It also sets the current device to the GPU
    corresponding to the provided rank.

    Parameters:
        rank (int): The rank of the current process.
        world_size (int): Total number of processes (GPUs) participating.
    """

    os.environ['MASTER_ADDR'] = 'localhost'

    # initialize the process group
    dist.init_process_group("nccl", "env://", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

# FUNCTION ENDING THE PROCESS
def cleanup():
    """
    Clean up the distributed process group.

    This function destroys the process group created for distributed training,
    ensuring that resources are properly released.
    """
    dist.destroy_process_group()


def run_VNA(rank: int, world_size: int,train_batch_size: int,key:str, num_layers:int, system_size: int,warmup_time: int, annealing_time: int, 
            equilibrium_time: int, num_units: int,weight_sharing:str,input_dim:int,train_size: int,warmup_on:str, annealing_on:str, temp_scheduler, optimizer_type:str, scheduler_name:str,
            ftype:torch.dtype, learning_rate, seed, T0,Tf,J_matrix,gather_interval):

    """
    Run the Variational Neural Ansatz (VNA) training process in a distributed setting.

    This function is intended to be run on each GPU process. It sets up the distributed training 
    environment, initializes seeds, defines the annealing schedule, prepares the training data,
    creates the ansatz model and the spin model, and then performs training through a trainer class.
    
    After training is complete, the distributed process group is cleaned up.

    Parameters:
        rank (int): Rank of the current GPU process.
        world_size (int): Total number of GPU processes.
        train_batch_size (int): Batch size for training on each process.
        key (str): Specifies the type of RNN cell (e.g., "vanilla" or "gru") for the ansatz model.
        num_layers (int): Number of RNN layers per site.
        system_size (int): Total number of spins (sites) in the system.
        warmup_time (int): Number of steps in the warmup phase.
        annealing_time (int): Number of annealing steps.
        equilibrium_time (int): Frequency (in steps) to update the annealing temperature.
        num_units (int): Number of units per RNN cell.
        input_dim: Dimensionality of the input (e.g., 2 for binary input).
        train_size (int): Total training size (number of samples).
        warmup_on (str): Whether warmup is enabled ("True" or "False").
        annealing_on (str): (Not explicitly used in the function, but may indicate annealing behavior).
        temp_scheduler: Scheduler type for annealing temperature ("exponential", "linear", or "quadratic").
        optimizer_type (str): Optimizer type ("adam", "rmsprop", "sgd").
        scheduler_name (str): Learning rate scheduler type (e.g., "StepLR", "CosineAnnealingLR", etc.).
        ftype (torch.dtype): Data type for tensors.
        learning_rate: Learning rate for the optimizer.
        seed: Base random seed for reproducibility.
        T0 (float): Initial temperature.
        Tf (float): Final temperature.
        J_matrix: Coupling matrix for the spin model.
        gather_interval (int): Interval (in epochs) at which to gather statistics from all GPUs.
    """

    setup(rank, world_size)
    device = torch.device(f'cuda:{rank}')
    seed_everything(seed, rank)
    stop_time = annealing_time*equilibrium_time + warmup_time + 1
    stop_time_brute_force = 500

    print("total_epoch=",stop_time)

    print("N=",system_size)

    print("Tf=",Tf)
    #define annealing schedule
    Temperature_list = set_annealing_schedule(warmup_on, temp_scheduler, warmup_time, annealing_time, equilibrium_time, T0, Tf, ftype)

    #prepare train data
    train_dataset = input_data(train_size,input_dim,ftype)
    train_data = prepare_dataloader(train_dataset, world_size, rank, train_batch_size)

    #initialize model and optimizer
    ansatz = model_ansatz(key,system_size, input_dim,num_layers,ftype,num_units=num_units,weight_sharing=weight_sharing,device=device)
    optimizer = optimizer_init(ansatz,learning_rate,optimizer_type)
    scheduler = scheduler_init(optimizer, stop_time, scheduler_name=scheduler_name)

    model = model_class(system_size,J_matrix,device,ftype)

    #VNA training
    #trainer = VNA_trainer(ansatz,train_data,optimizer,scheduler,model,rank)
    #meanE, meanM = trainer.train(stop_time, Temperature_list,gather_interval)

    #Brute force training at temperature=Tf
    trainer = Brute_Gradient_Descent(ansatz,train_data,optimizer,scheduler,model,rank)
    meanE, meanM = trainer.train(stop_time_brute_force,Tf,gather_interval)

    #print(ansatz.parameters())

    cleanup()