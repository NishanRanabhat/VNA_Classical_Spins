# Dynamical Finite-Size Scaling Analysis

A Python module for extracting critical exponents from finite-size scaling data. Finds optimal exponents that collapse data from different system sizes onto a universal master curve.

---

## Overview

Given measurements at different system sizes L, the module fits the scaling ansatz:

```
x_rescaled = x · L^(-a)
y_rescaled = y · L^b
```

Optimizes exponents `a` and `b` to minimize scatter around a spline fit through the collapsed data.

---

## Project Structure

```
Dynamical_FSS/
├── data_set.py       # ScalingDataset container class
├── dynamical_fss.py  # FiniteSizeScaling optimizer
└── run.py            # Example usage
```

| File | Purpose |
|------|---------|
| `data_set.py` | Simple container holding (L, x, y, err) for one system size |
| `dynamical_fss.py` | Core class: rescales data, fits spline, optimizes exponents |
| `run.py` | Example showing how to use the module |

---

## Requirements

```
numpy
scipy
```

---

## Usage

### 1. Prepare your data

For each system size, create a `ScalingDataset`:

```python
from data_set import ScalingDataset

# System size L=10
ds1 = ScalingDataset(
    system_size=10,
    x=np.array([...]),    # e.g., annealing times
    y=np.array([...]),    # e.g., residual energy
    err=np.array([...])   # uncertainties in y
)

# System size L=20
ds2 = ScalingDataset(system_size=20, x=..., y=..., err=...)

# System size L=40
ds3 = ScalingDataset(system_size=40, x=..., y=..., err=...)
```

### 2. Run the fit

```python
from dynamical_fss import FiniteSizeScaling

fss = FiniteSizeScaling(
    ds1, ds2, ds3,      # pass all datasets
    a0=1.0,             # initial guess for exponent a
    b0=0.5              # initial guess for exponent b
)

a, b = fss.fit()
print(f"Exponents: a = {a:.4f}, b = {b:.4f}")
```

### 3. Get the master curve (optional)

```python
spline = fss.get_spline()  # returns scipy UnivariateSpline
```

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `a0` | 1.0 | Initial guess for exponent a |
| `b0` | 0.5 | Initial guess for exponent b |
| `s_factor` | 1.0 | Spline smoothing factor multiplier |
| `k` | 3 | Spline degree |
| `method` | `'Nelder-Mead'` | Scipy optimizer method |
| `maxiter` | 1000 | Maximum optimizer iterations |

---

## Example

```python
import numpy as np
from data_set import ScalingDataset
from dynamical_fss import FiniteSizeScaling

# Mock data for three system sizes
L_values = [10, 20, 40]
datasets = []

for L in L_values:
    x = np.linspace(1, 100, 50)
    y = (x ** 0.5) * (L ** -0.25)  # fake scaling behavior
    err = y * 0.01
    datasets.append(ScalingDataset(L, x, y, err))

# Fit
fss = FiniteSizeScaling(*datasets, a0=0.5, b0=0.3)
a, b = fss.fit()

print(f"a = {a:.4f}, b = {b:.4f}")
```

---

## How It Works

1. **Flatten** all datasets into single arrays
2. **Rescale** using trial exponents (a, b) and take log
3. **Fit spline** through rescaled data
4. **Compute cost** as mean squared deviation from spline
5. **Optimize** exponents to minimize cost

The fitting is done in log-space with standardization for numerical stability.

---

## Typical Use Case

After running VNA at multiple system sizes, collect:
- `x` = annealing steps
- `y` = residual energy per spin
- `err` = standard error

Then use this module to extract dynamical critical exponents characterizing how the annealing time scales with system size.
