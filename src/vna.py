import torch
import numpy as np
import random
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from utilities import set_annealing_schedule, model_ansatz, model_class, optimizer_init, scheduler_init, seed_everything, input_data, prepare_dataloader
from trainer import VNA_trainer
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
    os.environ['MASTER_ADDR'] = 'localhost'

    # initialize the process group
    dist.init_process_group("nccl", "env://", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

# FUNCTION ENDING THE PROCESS
def cleanup():
    dist.destroy_process_group()


def run_VNA(rank: int, world_size: int,train_batch_size: int,key:str, num_layers:int, system_size: int,warmup_time: int, annealing_time: int, 
            equilibrium_time: int, num_units: int, input_dim,train_size: int,warmup_on:str, annealing_on:str, temp_scheduler, optimizer_type:str, scheduler_name:str,
            ftype:torch.dtype, learning_rate, seed, T0,Tf,J_matrix,gather_interval):

    setup(rank, world_size)
    device = torch.device(f'cuda:{rank}')
    seed_everything(seed, rank)
    stop_time = annealing_time*equilibrium_time + warmup_time + 1

    #define annealing schedule
    Temperature_list = set_annealing_schedule(warmup_on, temp_scheduler, warmup_time, annealing_time, equilibrium_time, T0, Tf, ftype)

    #prepare train data
    train_dataset = input_data(train_size,input_dim,ftype)
    train_data = prepare_dataloader(train_dataset, world_size, rank, train_batch_size)

    #initialize model and optimizer
    ansatz = model_ansatz(key,system_size, input_dim,num_layers,ftype,num_units=num_units, seed=seed, device=device)
    optimizer = optimizer_init(ansatz,learning_rate,optimizer_type)
    scheduler = scheduler_init(optimizer, stop_time, scheduler_name=scheduler_name)

    model = model_class(system_size,J_matrix,device,ftype)

    trainer = VNA_trainer(ansatz,train_data,optimizer,scheduler,model,rank)
    meanE = trainer.train(stop_time, Temperature_list,gather_interval)

    cleanup()








