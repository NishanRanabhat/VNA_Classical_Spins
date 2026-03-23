import torch
import numpy as np
import os
import json
import sys
from pathlib import Path
from vna import run_VNA
from interaction_matrix import Sherrington_Kirkpatrick_1D, Fully_connected_1D, Nearest_neighbor_1D
import time
import torch.multiprocessing as mp
from multiprocessing import Manager

if "MASTER_PORT" not in os.environ:
    os.environ["MASTER_PORT"] = "12355" 

if __name__ == "__main__":
    
    wd = os.getcwd() 
    parameters_filename = Path(wd + "/input_files/simulations_parameters_q.json")
    # parameters_filename = Path(wd + "/input_files/simulations_parameters.json")
    with open(parameters_filename, 'r') as openfile:
        parameters = json.load(openfile)
    
    # Get system_size from command-line arguments
    seed = parameters['seed']
    T0 = parameters['initial_temperature']
    Tf = parameters['final_temperature']
    system_size = parameters['system_size']
    input_dim = parameters['input_dim']
    eq_trainer = parameters["eq_trainer"]
    annealing_on = parameters['annealing_on']
    warmup_on = parameters['warmup_on']
    rnn_type = parameters['rnn_type']
    temp_scheduler = parameters['temperature_scheduler']
    optimizer = parameters['optimizer']
    scheduler_name = parameters['lr_scheduler']
    num_samples = parameters['num_samples']
    num_units = parameters['num_units']
    learning_rate = parameters['learning_rate']
    weight_sharing = parameters['weight_sharing']
    num_layers = parameters['num_layers']
    equilibrium_time = parameters['equilibration_time']
    warmup_time = parameters['warmup_time']
    interaction_name = parameters['interaction']
    # annealing_time = parameters['annealing_time']
    annealing_time = int(sys.argv[1])

    if interaction_name == "nearest_neighbor":
        J_matrix = Nearest_neighbor_1D(system_size)
    elif interaction_name == "fully_connected":
        J_matrix = Fully_connected_1D(system_size)
    elif interaction_name == "sherrington_kirkpatrick":
        J_matrix = Sherrington_Kirkpatrick_1D(system_size)
    else:
        error_message = ("Invalid interaction type specified in parameters"
            f": {parameters['interaction']}. Valid options are 'nearest_neighbor',"
            " 'fully_connected', or 'sherrington_kirkpatrick'.")
        raise ValueError(error_message)
    
    ftype = torch.float32

    world_size = torch.cuda.device_count()
    train_batch_size = int(num_samples / world_size)
    sample_batch_size = train_batch_size
    train_size = train_batch_size * world_size

    print(f"Running simulation with system_size={system_size}")
    print("Train_size=", train_size)
    print("annealing_steps", annealing_time * equilibrium_time + warmup_time)

    tic = time.time()
    gather_interval = 1  # parameters['equilibration_time']

    mp.spawn(run_VNA, args=(world_size, eq_trainer, train_batch_size, 
                            rnn_type, num_layers, 
                            system_size, warmup_time, annealing_time, equilibrium_time, 
                            num_units, weight_sharing, input_dim, train_size, 
                            warmup_on, annealing_on,
                            temp_scheduler, optimizer, 
                            scheduler_name, interaction_name, ftype, learning_rate, seed, T0, Tf,
                            J_matrix, gather_interval), nprocs=world_size, join=True)
    

    tac = time.time()
    print("total time = ", tac - tic)