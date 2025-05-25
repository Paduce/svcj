# Drift Signal Preservation for μ Estimation

## Problem
The original code was doing **per-series centering** which removes the drift signal that encodes μ:

```python
# PROBLEMATIC: This removes drift signal!
if normalize_returns:
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
```

## Solution
Switch to **global normalization** to preserve cross-sample μ differences:

### 1. Dataset Configuration
```python
config = TrainingConfig(
    normalize_returns=False,    # NO per-series centering in dataset
    standardize_params=False,   # NO per-param norm in dataset
    # ... other parameters
)
```

### 2. Training Script Behavior
- When `normalize_returns=False`: Computes global statistics over ALL training data
- Applies global normalization: `(x - global_mean) / global_std`  
- Preserves relative differences between samples that encode μ

### 3. Key Changes Made

#### In `datasets.py`:
- Added clear warnings about per-series centering removing drift
- Updated default values to preserve drift signal
- Added detailed comments explaining normalization trade-offs

#### In `train.py`:
- Enhanced logging to clearly indicate when drift is preserved vs. lost
- Added comprehensive docstring explaining proper usage
- Updated example configuration to show recommended settings

### 4. How It Works

**Bad (per-series centering):**
```
Sample 1: [0.02, 0.01, 0.03] → [0.0, -0.01, 0.01]  # μ lost!
Sample 2: [0.05, 0.04, 0.06] → [0.0, -0.01, 0.01]  # μ lost!
```

**Good (global normalization):**
```
Global mean = 0.035
Sample 1: [0.02, 0.01, 0.03] → [-0.15, -0.25, -0.05]  # μ preserved!
Sample 2: [0.05, 0.04, 0.06] → [0.15, 0.05, 0.25]     # μ preserved!
```

### 5. Usage
```python
from src.svcj.dl.train import TrainingConfig, train

# Correct configuration for μ estimation
config = TrainingConfig(
    model_type="svcj",
    mu_mode="raw",              # Include μ in predictions
    normalize_returns=False,    # Preserve drift signal
    standardize_params=False,   # Use global param normalization
    seq_len=672,
    n_samples=100_000,
    # ... other parameters
)

model = train(config)  # Drift signal preserved!
```

The model can now learn to map return sequences to μ because the cross-sample drift differences are preserved in the data. 