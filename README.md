# Variational Neural Annealing for Classical Spin Models

A PyTorch implementation of Variational Neural Annealing (VNA) using recurrent neural networks to find ground states of classical spin Hamiltonians. The method variationally simulates classical annealing, avoiding slow Markov chain dynamics by leveraging autoregressive sampling from RNNs.

**Reference:** [Hibat-Allah et al., Nature Machine Intelligence (2021)](https://www.nature.com/articles/s42256-021-00401-3)

---

## Project Structure

```
├── src/
│   ├── main.py                  # Entry point — launches distributed training
│   ├── vna.py                   # Training orchestration (setup, run, cleanup)
│   ├── trainer.py               # Training loops (VNA_trainer, Brute_Gradient_Descent)
│   ├── neural_network_ansatz.py # RNN wavefunction architectures
│   ├── objective_function.py    # Spin model energy/magnetization computation
│   ├── interaction_matrix.py    # Coupling matrices (SK, fully-connected, nearest-neighbor)
│   ├── utilities.py             # Annealing schedules, initializers, data loaders
│   └── exact_free_energy_1D_NN_Ising.py  # Analytical benchmarks for 1D Ising
│
├── input_files/
│   └── simulations_parameters.json  # Configuration file
│
└── models/                      # Saved model checkpoints (created at runtime)
```

### File Descriptions

| File | Purpose | Modify when... |
|------|---------|----------------|
| `main.py` | Reads config, sets up interaction matrix, spawns multi-GPU processes | Changing which interaction matrix to use |
| `vna.py` | Sets up distributed training, initializes all components, runs training loop | Switching between VNA and brute-force training |
| `trainer.py` | Contains `VNA_trainer` (annealing) and `Brute_Gradient_Descent` (fixed T) classes | Changing loss function or training logic |
| `neural_network_ansatz.py` | RNN architectures: with and without weight sharing | Adding new network architectures |
| `objective_function.py` | `One_dimensional_spin_model`: converts samples to spins, computes energy | Adding new observables |
| `interaction_matrix.py` | Generates J matrices: Sherrington-Kirkpatrick, fully-connected, nearest-neighbor | Adding new spin models |
| `utilities.py` | Annealing schedules, optimizer/scheduler init, seeding, data prep | Adding new schedulers or optimizers |
| `exact_free_energy_1D_NN_Ising.py` | Exact free energy formulas for 1D Ising (finite-N and thermodynamic limit) | Benchmarking against exact solutions |

---

## Requirements

```
torch >= 2.0
numpy
```

**Hardware:** Multi-GPU recommended (uses `torch.distributed` with NCCL backend). Falls back to single GPU if only one is available.

---

## Configuration

Edit `input_files/simulations_parameters.json`:

```json
{
    "system_size"           : 50,
    "input_dim"             : 2,
    "num_samples"           : 5000,
    "num_units"             : 30,
    "num_layers"            : 2,
    "equilibration_time"    : 5,
    "warmup_time"           : 50,
    "annealing_time"        : 200,
    "initial_temperature"   : 1.2,
    "final_temperature"     : 0.001,
    "weight_sharing"        : "True",
    "annealing_on"          : "True",
    "warmup_on"             : "True",
    "rnn_type"              : "gru",
    "optimizer"             : "adam",
    "temperature_scheduler" : "exponential",
    "lr_scheduler"          : "Exponential"
}
```

### Parameter Reference

| Parameter | Description |
|-----------|-------------|
| `system_size` | Number of spins (N) |
| `input_dim` | Input dimension (2 for binary spins) |
| `num_samples` | Samples per GPU per epoch |
| `num_units` | Hidden units in RNN cells |
| `num_layers` | Stacked RNN layers |
| `equilibration_time` | Training steps per temperature |
| `warmup_time` | Steps at initial temperature before annealing |
| `annealing_time` | Number of temperature steps |
| `initial_temperature` | Starting temperature (T₀) |
| `final_temperature` | Target temperature (T_f) |
| `weight_sharing` | `"True"`: single RNN for all sites; `"False"`: separate RNN per site |
| `rnn_type` | `"gru"` or `"vanilla"` |
| `optimizer` | `"adam"`, `"rmsprop"`, or `"sgd"` |
| `temperature_scheduler` | `"exponential"`, `"linear"`, or `"quadratic"` |
| `lr_scheduler` | `"Exponential"`, `"StepLR"`, `"CosineAnnealingLR"`, `"None"`, etc. |

**Total epochs** = `warmup_time` + `annealing_time` × `equilibration_time`

---

## Running the Code

1. **Create required directories:**
   ```bash
   mkdir -p input_files models
   ```

2. **Place configuration file** in `input_files/simulations_parameters.json`

3. **Run:**
   ```bash
   cd src
   python main.py
   ```

The code automatically detects available GPUs and distributes training across them.

---

## Changing the Spin Model

In `main.py`, modify the `J_matrix` line:

```python
# Fully-connected (mean-field)
J_matrix = Fully_connected_1D(system_size)

# Nearest-neighbor 1D Ising
J_matrix = Nearest_neighbor_1D(system_size)

# Sherrington-Kirkpatrick (random Gaussian couplings)
J_matrix = Sherrington_Kirkpatrick_1D(system_size)
```

---

## Training Modes

In `vna.py`, the `run_VNA` function has two training options:

```python
# Option 1: VNA with annealing schedule
trainer = VNA_trainer(ansatz, train_data, optimizer, scheduler, model, rank)
meanE, meanM = trainer.train(stop_time, Temperature_list, gather_interval)

# Option 2: Brute-force optimization at fixed temperature
trainer = Brute_Gradient_Descent(ansatz, train_data, optimizer, scheduler, model, rank)
meanE, meanM = trainer.train(stop_time_brute_force, Tf, gather_interval)
```

Currently, brute-force mode is active. Comment/uncomment to switch.

---

## Output

- **Console:** Energy and magnetization printed each epoch
- **Model checkpoint:** Saved to `models/model_{temperature}.pt`

---

## Key Concepts

**Variational free energy minimization:**
```
F_loc = E_loc + T × log(p)
```
where `E_loc` is the energy of a sampled configuration and `log(p)` is the log-probability from the RNN.

**Loss function (REINFORCE-style gradient):**
```
cost = ⟨log(p) × F_loc⟩ - ⟨log(p)⟩ × ⟨F_loc⟩
```

**Annealing:** Temperature decreases from T₀ to T_f following the chosen schedule, guiding the RNN distribution toward low-energy configurations.

---

## Extending the Code

- **New interaction matrix:** Add function in `interaction_matrix.py`, return upper-triangular NumPy array
- **New observable:** Add method to `One_dimensional_spin_model` in `objective_function.py`
- **New RNN architecture:** Add class in `neural_network_ansatz.py`, ensure it has `.samples` and `.log_probs` attributes after forward pass

---

## Dynamical Finite-Size Scaling

This project also includes an independent module for **dynamical finite-size scaling analysis** in `Dynamical_FSS/`. Use it to extract critical exponents from VNA results at multiple system sizes.

See [`Dynamical_FSS/Dynamical_FSS_README.md`](Dynamical_FSS/Dynamical_FSS_README.md) for usage instructions.
