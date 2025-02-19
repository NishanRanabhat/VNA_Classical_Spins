import torch
import numpy as np
import os
import json
from pathlib import Path
from vna import run_VNA
from interaction_matrix import Sherrington_Kirkpatrick_1D, Fully_connected_1D
import time
import torch.multiprocessing as mp
from multiprocessing import Manager

if __name__ == "__main__":
    
    wd = os.getcwd() 

    parameters_filename = Path( wd + "/input_files/simulations_parameters.json")
    with open(parameters_filename, 'r') as openfile:
        parameters = json.load(openfile)
    
    system_size = parameters['system_size'] 
    input_dim = parameters['input_dim']

    annealing_on = parameters['annealing_on']
    warmup_on = parameters['warmup_on']
    rnn_type = parameters['rnn_type']
    temp_scheduler = parameters['temperature_scheduler']
    optimizer = parameters['optimizer']
    scheduler_name = parameters['lr_scheduler']

    num_samples = parameters['num_samples']
    num_units = parameters['num_units']
    num_layers = parameters['num_layers']
    equilibrium_time = parameters['equilibration_time']
    warmup_time = parameters['warmup_time']
    annealing_time = parameters['annealing_time']

    T0=parameters['initial_temperature']
    Tf=parameters['final_temperature']

    J_matrix = Fully_connected_1D(system_size)

    learning_rate = 2e-3
    seed = 12345
    ftype= torch.float32

    world_size = torch.cuda.device_count()
    train_batch_size = int(num_samples)
    sample_batch_size = train_batch_size
    train_size = train_batch_size*world_size

    print("Train_size=",train_size)
    print("annealing_steps",annealing_time*equilibrium_time+warmup_time)

    tic = time.time()
    gather_interval = parameters['equilibration_time']

    mp.spawn(run_VNA,args=(world_size,train_batch_size,rnn_type, num_layers, system_size,warmup_time, annealing_time, equilibrium_time, num_units,
                       input_dim,train_size,warmup_on, annealing_on, temp_scheduler, optimizer, scheduler_name, ftype,learning_rate, seed, T0,Tf,
                       J_matrix,gather_interval),nprocs=world_size,join=True)

    tac = time.time()
    print("total time = ",tac-tic)

