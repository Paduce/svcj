import sys
from pathlib import Path

import numpy as np
import torch

from svcj.dl.models import build_model
from svcj.dl.config import get_param_order, get_param_ranges

DEFAULT_SEQ_LEN = 672  # four weeks of hourly data
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def predict(
    model_path,
    arch="cnn",
    model_type="svcj",
    input_path=None,
    device="cpu",
    return_series=None,
    seq_len: int = DEFAULT_SEQ_LEN,
    agg: str = "median",
    mu_mode="raw",
    TRAIN_X_MEAN=0.00012,
    TRAIN_X_STD=0.13,
):
    """
    Run model prediction with explicit arguments instead of argparse.
    """
    # 1. Load input data
    if return_series is not None:
        vec = return_series
    elif input_path in (None, "-"):
        print("Reading input vector from stdin (comma-separated floats):", file=sys.stderr)
        vec = np.fromstring(sys.stdin.read(), sep=",")
    else:
        vec = np.load(input_path)
    
    vec = vec.astype(np.float32).ravel()
    if vec.size < seq_len:
        raise ValueError(f"Need at least {seq_len} observations (got {vec.size}).")

    # Cut to an integer number of windows and reshape → (B, seq_len)
    n_win = vec.size // seq_len
    x = vec[: n_win * seq_len].reshape(n_win, seq_len)
    sample_mean = vec.mean()
    x = torch.from_numpy(x).to(device)

    # 2. Standardise each window independently (same as training)
    param_order = get_param_order(model_type)
    param_ranges = get_param_ranges(model_type)
    lows  = np.array([param_ranges[k].low  for k in param_order], dtype=np.float32)
    highs = np.array([param_ranges[k].high for k in param_order], dtype=np.float32)

    # Use the same standardization constants from training
    x_mean = torch.tensor(TRAIN_X_MEAN, device=device)
    x_std  = torch.tensor(TRAIN_X_STD, device=device)
    x_norm = (x - x_mean) / (x_std + 1e-8)   # (B, seq_len)

    # 3. Load model
    model = build_model(arch, model_type=model_type).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 4. Predict normalised parameters
    with torch.no_grad():
        if mu_mode == "raw":
            pred_norm = model(x_norm).cpu().numpy()
        else:
            pred_norm = model(x_norm, mu_mode).cpu().numpy()   # (B, n_params)

    if agg == "mean":
        pred_norm = pred_norm.mean(axis=0)
    elif agg == "median":
        pred_norm = np.median(pred_norm, axis=0)
    else:
        raise ValueError("agg must be 'mean' or 'median'")

    # 5. De-normalise output (reverse target normalization)
    y_mean = ((lows + highs) / 2)
    y_std  = ((highs - lows) / np.sqrt(12))

    pred = pred_norm * y_std + y_mean

    if mu_mode == "analytical":
        mu_hat = sample_mean / (1/8760.0)   # dt⁻¹
        pred = np.insert(pred, 0, mu_hat)

    # 6. Collect results
    return dict(zip(param_order, pred))
