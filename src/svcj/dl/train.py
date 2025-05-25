# ───────────────────────────────────────────────────────────────────────────
# sv_family_training.py
# Complete, coherent training procedure for the SV-family 1D-CNN
# Integrated with enhanced simulation and dataset modules
# ---------------------------------------------------------------------------
import os
import random
import warnings
from statistics import mean
from typing import Optional, Dict, Any, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm

# Import enhanced modules
from svcj.dl.datasets import (
    SVFamilyDataset, CachedSVDataset, BatchSVDataset, 
    DatasetConfig, loader, create_validation_dataset, analyze_dataset
)
from svcj.dl.simulator import SVParams, validate_parameters, DT_HOUR
from svcj.dl.models import build_model                      # unchanged
from svcj.dl.logger import TBLogger                         # unchanged
from svcj.dl.config import get_param_order, get_param_ranges

# deterministic cuDNN kernels are slower, but give repeatable results
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

__all__ = ["train", "TrainingConfig", "validate_model", "estimate_input_norm"]


# ───────────────────────────────────────────────────────────────────────────
# Training configuration
# ---------------------------------------------------------------------------
class TrainingConfig:
    """Comprehensive training configuration."""
    
    def __init__(
        self,
        # Model architecture
        arch: str = "cnn",
        model_type: str = "svcj",
        
        # Data configuration
        seq_len: int = 672,
        n_samples: int = 110_000,
        val_split: float = 0.1,
        dt: float = DT_HOUR,
        rho: Optional[float] = None,
        
        # Training hyperparameters
        epochs: int = 100,
        batch_size: int = 512,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        
        # Training behavior
        mu_mode: str = "raw",
        mixed_precision: bool = True,
        early_stop_patience: int = 6,
        validate_every: int = 1,
        clip_predictions: bool = True,
        
        # Data options - IMPORTANT: preserve drift signal for μ estimation
        normalize_returns: bool = False,    # False = preserve drift, use global norm in training
        standardize_params: bool = False,   # False = use global param norm in training
        return_variance: bool = False,
        return_jumps: bool = False,
        use_cached_dataset: bool = False,
        cache_path: Optional[str] = None,
        
        # System configuration
        device: Optional[str] = None,
        num_workers: int = 4,
        seed: int = 42,
        logdir: str = "runs/sv_family",
        load_checkpoint: bool = False,
    ):
        # Store all parameters
        self.arch = arch
        self.model_type = model_type.lower()
        self.seq_len = seq_len
        self.n_samples = n_samples
        self.val_split = val_split
        self.dt = dt
        self.rho = rho
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.mu_mode = mu_mode.lower()
        self.mixed_precision = mixed_precision
        self.early_stop_patience = early_stop_patience
        self.validate_every = validate_every
        self.clip_predictions = clip_predictions
        self.normalize_returns = normalize_returns
        self.standardize_params = standardize_params
        self.return_variance = return_variance
        self.return_jumps = return_jumps
        self.use_cached_dataset = use_cached_dataset
        self.cache_path = cache_path
        self.device = device
        self.num_workers = num_workers
        self.seed = seed
        self.logdir = logdir
        self.load_checkpoint = load_checkpoint
        
        self.validate()
    
    def validate(self):
        """Validate configuration parameters."""
        if self.model_type not in ("sv", "svj", "svcj"):
            raise ValueError(f"model_type must be one of ['sv', 'svj', 'svcj'], got {self.model_type}")
        if self.mu_mode not in ("raw", "analytical", "devol"):
            raise ValueError(f"mu_mode must be one of ['raw', 'analytical', 'devol'], got {self.mu_mode}")
        if not 0 < self.val_split < 1:
            raise ValueError(f"val_split must be in (0, 1), got {self.val_split}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        if self.rho is not None and not -1 <= self.rho <= 1:
            raise ValueError(f"rho must be in [-1, 1], got {self.rho}")


# ───────────────────────────────────────────────────────────────────────────
# Utility functions
# ---------------------------------------------------------------------------
def set_global_seed(seed: int = 42) -> None:
    """Set global random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def estimate_input_norm(dl: DataLoader, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Single-pass computation of global mean / std of inputs using Welford's algorithm.
    
    Returns:
        (mean, std) tensors on the specified device
    """
    n = 0
    mean_x = torch.zeros(1, device=device)
    m2 = torch.zeros(1, device=device)  # Σ(x-μ)²

    for batch in tqdm(dl, desc="Estimating input normalization statistics"):
        xb = batch[0]  # First element is always returns
        xb = xb.to(device, non_blocking=True)
        batch_n = xb.numel()
        
        if batch_n == 0:
            continue
            
        n_new = n + batch_n

        # Welford's online algorithm
        delta = xb.mean() - mean_x
        mean_x += delta * batch_n / n_new
        m2 += ((xb - mean_x)**2).sum()
        n = n_new

    if n == 0:
        raise ValueError("No data found in input normalization")
        
    var = m2 / n
    return mean_x, var.sqrt().clamp_min(1e-8)


@torch.inference_mode()
def compute_target_statistics(dl: DataLoader, device: torch.device, 
                            mu_mode: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute target mean and std from a dataloader.
    
    Returns:
        (mean, std) tensors of shape (n_params,)
    """
    all_targets = []
    
    for batch in dl:
        yb = batch[1]  # Second element is always parameters
        yb = yb.to(device, non_blocking=True)
        # Note: mu_mode handling is already done in the dataset
        all_targets.append(yb)
    
    if not all_targets:
        raise ValueError("No targets found in dataloader")
    
    all_targets = torch.cat(all_targets, dim=0)
    
    mean_y = all_targets.mean(dim=0)
    std_y = all_targets.std(dim=0, unbiased=False).clamp_min(1e-8)
    
    return mean_y, std_y


def _standardise(x: torch.Tensor, mean_: torch.Tensor, std_: torch.Tensor) -> torch.Tensor:
    """Standardize tensor using provided mean and std."""
    return (x - mean_) / std_


@torch.inference_mode()
def compute_r2_incremental(model: nn.Module, dl_val: DataLoader, device: torch.device,
                          x_mean: torch.Tensor, x_std: torch.Tensor,
                          y_mean: torch.Tensor, y_std: torch.Tensor,
                          config: TrainingConfig) -> Tuple[torch.Tensor, float]:
    """
    Compute R² metrics incrementally to save memory.
    
    Returns:
        (r2_per_param, r2_global)
    """
    # Import here to avoid circular imports
    from svcj.dl.datasets import SVFamilyDataset, loader
    
    model.eval()
    
    ss_res = torch.zeros_like(y_mean)
    ss_tot = torch.zeros_like(y_mean)
    n_samples = 0
    val_loss_accum = 0.0
    loss_fn = nn.MSELoss(reduction="none")
    
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=config.mixed_precision):
        for batch in dl_val:
            xb, yb = batch[0], batch[1]  # Returns and parameters
            xb = _standardise(xb.to(device, non_blocking=True), x_mean, x_std)
            yb = yb.to(device, non_blocking=True)
            
            # Normalize targets
            yb_norm = (yb - y_mean) / y_std
            
            # Get predictions
            preds_norm = model(xb, config.mu_mode)
            
            # Ensure dimensions match
            if preds_norm.shape != yb_norm.shape:
                raise RuntimeError(
                    f"Prediction shape {preds_norm.shape} doesn't match "
                    f"target shape {yb_norm.shape}"
                )
            
            # Validation loss (normalized space)
            val_loss_accum += loss_fn(preds_norm, yb_norm).mean().item()
            
            # Denormalize predictions for R²
            preds = preds_norm * y_std + y_mean
            
            # Accumulate for R²
            ss_res += ((preds - yb) ** 2).sum(dim=0)
            ss_tot += ((yb - y_mean) ** 2).sum(dim=0)
            n_samples += yb.size(0)
    
    # Compute R² with numerical stability
    r2_per_param = 1.0 - ss_res / (ss_tot + 1e-8)
    r2_global = r2_per_param.mean().item()
    avg_val_loss = val_loss_accum / len(dl_val)
    
    return r2_per_param, r2_global, avg_val_loss


def validate_model_architecture(model: nn.Module, sample_input: torch.Tensor, 
                               expected_output_dim: int, mu_mode: str, 
                               device: torch.device) -> None:
    """Validate model architecture matches expected dimensions."""
    
    model.eval()
    with torch.no_grad():
        sample_output = model(sample_input.to(device), mu_mode)
        
    if sample_output.shape[1] != expected_output_dim:
        raise ValueError(
            f"Model output dimension mismatch: expected {expected_output_dim}, "
            f"got {sample_output.shape[1]}"
        )


def clip_sv_parameters(params: torch.Tensor, model_type: str, 
                      param_names: list) -> torch.Tensor:
    """Clip parameters to economically valid ranges."""
    
    if not hasattr(clip_sv_parameters, '_constraints'):
        clip_sv_parameters._constraints = {
            'mu': (-0.5, 0.5),
            'v_long': (1e-4, 1.0), 
            'theta': (1e-4, 1.0),   # alias for v_long
            'beta': (0.01, 0.99),
            'kappa': (0.01, 10.0),
            'gamma': (1e-4, 2.0),
            'sigma': (1e-4, 2.0),   # alias for gamma
            'rho': (-0.99, 0.99),
            'lambda': (0.0, 5.0),
            'lam': (0.0, 5.0),      # alias for lambda
            'mu_j': (-0.5, 0.5),
            'sigma_j': (1e-4, 1.0),
            'v0': (1e-4, 1.0),
        }
    
    params_clipped = params.clone()
    
    for i, name in enumerate(param_names):
        if name in clip_sv_parameters._constraints:
            min_val, max_val = clip_sv_parameters._constraints[name]
            params_clipped[:, i] = torch.clamp(params_clipped[:, i], min_val, max_val)
    
    return params_clipped


# ───────────────────────────────────────────────────────────────────────────
# Main training function
# ---------------------------------------------------------------------------
def train(config: Optional['TrainingConfig'] = None, **kwargs) -> nn.Module:
    """
    Train a 1-D CNN to map return sequences to SV model parameters.
    
    IMPORTANT: For μ (drift) estimation, use global normalization to preserve 
    the cross-sample drift signal:
    
    ```python
    config = TrainingConfig(
        normalize_returns=False,    # Preserve drift in dataset
        standardize_params=False,   # Use global param normalization
        # ... other parameters
    )
    model = train(config)
    ```
    
    This approach:
    - Preserves cross-sample mean differences that encode μ
    - Applies global normalization in training for numerical stability
    - Allows the model to see true drift variations across samples
    
    Parameters
    ----------
    config : TrainingConfig, optional
        Training configuration object. If None, created from kwargs.
    **kwargs
        Training parameters (used if config is None).
        
    Returns
    -------
    nn.Module
        Trained model.
    """
    
    # Create config if not provided
    if config is None:
        config = TrainingConfig(**kwargs)
    
    # Import dataset here to avoid circular imports
    from svcj.dl.datasets import SVFamilyDataset, DatasetConfig
    
    # ─── Setup logging and device ──────────────────────────────────────
    logger = TBLogger(config.logdir, log_to_file=True)
    logger.log_msg(f"==== SV-family Trainer started ({config.model_type.upper()}) ====")
    logger.log_msg(f"Training configuration: {config.__dict__}")

    set_global_seed(config.seed)
    
    if config.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.device)
    logger.log_msg(f"→ device: {device}")

    # ─── Model dimensions and parameter setup ──────────────────────────
    param_order = get_param_order(config.model_type)
    original_n_params = len(param_order)
    effective_n_params = original_n_params - 1 if config.mu_mode == "analytical" else original_n_params
    
    logger.log_msg(f"Parameters ({original_n_params} → {effective_n_params}): {param_order}")
    if config.mu_mode == "analytical":
        logger.log_msg("→ μ parameter excluded (analytical mode)")

    # ─── Dataset creation ───────────────────────────────────────────────
    try:
        # Create dataset configuration
        dataset_config = DatasetConfig(
            seq_len=config.seq_len,
            n_samples=config.n_samples,
            model_type=config.model_type,
            mu_mode=config.mu_mode,
            dt=config.dt,
            rho=config.rho,
            s0=1.0,
            v0=None,
            return_variance=config.return_variance,
            return_jumps=config.return_jumps,
            normalize_returns=config.normalize_returns,
            standardize_params=config.standardize_params,
            seed=config.seed,
            deterministic=True,
            cache_data=config.use_cached_dataset,
            cache_path=config.cache_path,
        )
        
        # Create dataset
        if config.use_cached_dataset:
            logger.log_msg("Using cached dataset...")
            ds_full = CachedSVDataset(
                cache_path=config.cache_path,
                config=dataset_config
            )
        else:
            logger.log_msg("Using on-the-fly dataset generation...")
            ds_full = SVFamilyDataset(config=dataset_config)
        
        # Analyze dataset
        logger.log_msg("Analyzing dataset...")
        analysis = analyze_dataset(ds_full, n_samples=min(100, len(ds_full)))
        logger.log_msg(f"Dataset analysis: success_rate={analysis['success_rate']:.2%}")
        
    except Exception as e:
        logger.log_msg(f"Error creating dataset: {e}", level=40)
        raise
    
    # ─── Train/validation split ─────────────────────────────────────────
    n_val = int(len(ds_full) * config.val_split)
    n_train = len(ds_full) - n_val
    
    if n_train <= 0 or n_val <= 0:
        raise ValueError(f"Invalid dataset split: train={n_train}, val={n_val}")
    
    # Use different seeds for train/val to ensure different parameter sampling
    train_config_dict = {k: v for k, v in dataset_config.__dict__.items()}
    train_config_dict['n_samples'] = n_train
    train_config_dict['seed'] = config.seed  # Use training seed
    
    val_config_dict = {k: v for k, v in dataset_config.__dict__.items()}
    val_config_dict['n_samples'] = n_val
    val_config_dict['seed'] = config.seed + 1000  # Different seed for validation
    
    ds_train = SVFamilyDataset(config=DatasetConfig(**train_config_dict))
    ds_val = SVFamilyDataset(config=DatasetConfig(**val_config_dict))
    
    dl_train = loader(ds_train, batch_size=config.batch_size, shuffle=True,
                      num_workers=config.num_workers, pin_memory=True)
    dl_val = loader(ds_val, batch_size=config.batch_size, shuffle=False,
                    num_workers=config.num_workers, pin_memory=True)
    
    logger.log_msg(f"Dataset ready · train={n_train}, val={n_val}")

    # ─── Verify data dimensions ─────────────────────────────────────────
    try:
        sample_batch = next(iter(dl_train))
        sample_returns, sample_params = sample_batch[0], sample_batch[1]
        
        logger.log_msg(f"Data shapes: returns={sample_returns.shape}, params={sample_params.shape}")
        
        if sample_params.shape[1] != effective_n_params:
            raise ValueError(
                f"Parameter dimension mismatch: expected {effective_n_params}, "
                f"got {sample_params.shape[1]}"
            )
        
        # Check for additional outputs
        n_outputs = len(sample_batch)
        output_info = [f"returns{sample_returns.shape}", f"params{sample_params.shape}"]
        
        if config.return_variance and n_outputs > 2:
            output_info.append(f"variance{sample_batch[2].shape}")
        if config.return_jumps and n_outputs > (3 if config.return_variance else 2):
            output_info.append(f"jumps{sample_batch[-1].shape}")
        
        logger.log_msg(f"Dataset outputs: {', '.join(output_info)}")
        
    except Exception as e:
        logger.log_msg(f"Error validating data dimensions: {e}", level=40)
        raise

    # ─── Input and target normalization ─────────────────────────────────
    # CRITICAL: Use global normalization to preserve drift signal that encodes μ
    try:
        # Global input normalization - preserves cross-sample μ differences
        if not config.normalize_returns:  # Recommended for μ estimation
            logger.log_msg("Computing global input normalization (preserves drift signal)...")
            x_mean, x_std = estimate_input_norm(
                loader(ds_train, batch_size=min(2048, n_train), shuffle=False, 
                      num_workers=config.num_workers, pin_memory=True), 
                device
            )
            logger.log_msg(f"Global input stats: μ={x_mean.item():.4g}, σ={x_std.item():.4g}")
            logger.log_msg("✓ Cross-sample drift signal preserved for μ estimation")
        else:
            # Per-series normalization was applied in dataset - drift signal lost!
            x_mean, x_std = torch.tensor(0.0, device=device), torch.tensor(1.0, device=device)
            logger.log_msg("⚠ Returns pre-normalized per-series - drift signal removed!")
        
        # Global parameter normalization
        if not config.standardize_params:  # Recommended approach
            logger.log_msg("Computing global parameter normalization...")
            y_mean, y_std = compute_target_statistics(
                loader(ds_train, batch_size=min(4096, n_train), shuffle=False,
                      num_workers=config.num_workers, pin_memory=True),
                device, config.mu_mode
            )
            logger.log_msg(f"Global param stats: μ ∈ [{y_mean.min():.3f}, {y_mean.max():.3f}], "
                          f"σ ∈ [{y_std.min():.3f}, {y_std.max():.3f}]")
        else:
            # Parameters were already standardized in dataset
            y_mean = torch.zeros(effective_n_params, device=device)
            y_std = torch.ones(effective_n_params, device=device)
            logger.log_msg("Parameters pre-standardized in dataset")
            
    except Exception as e:
        logger.log_msg(f"Error computing normalization statistics: {e}", level=40)
        raise

    # ─── Model creation and validation ──────────────────────────────────
    try:
        model = build_model(config.arch, model_type=config.model_type).to(device)
        
        # Validate model architecture
        validate_model_architecture(
            model, sample_returns[:1], effective_n_params, 
            config.mu_mode, device
        )
        
        n_params = sum(p.numel() for p in model.parameters())
        logger.log_msg(f"Model created: {n_params:,} parameters")
        
    except Exception as e:
        logger.log_msg(f"Error creating model: {e}", level=40)
        raise

    # ─── Model initialization ───────────────────────────────────────────
    def _init_weights(m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    model.apply(_init_weights)

    # ─── Optimizer setup ────────────────────────────────────────────────
    # Separate parameter groups for different weight decay
    backbone_params = []
    log_var_params = []
    
    for name, param in model.named_parameters():
        if 'log_var' in name.lower():
            log_var_params.append(param)
        else:
            backbone_params.append(param)
    
    param_groups = [{"params": backbone_params, "weight_decay": config.weight_decay}]
    if log_var_params:
        param_groups.append({"params": log_var_params, "weight_decay": 0.0})
        logger.log_msg(f"Parameter groups: backbone ({len(backbone_params)}), "
                      f"log_vars ({len(log_var_params)})")
    
    opt = Adam(param_groups, lr=config.lr)
    sched = ReduceLROnPlateau(opt, mode="min", factor=0.1, patience=3, verbose=True)
    loss_fn = nn.MSELoss(reduction="none")
    scaler = torch.cuda.amp.GradScaler(enabled=config.mixed_precision)

    # ─── Checkpoint handling ────────────────────────────────────────────
    ckpt_dir = os.path.join(config.logdir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_file = os.path.join(ckpt_dir, "best.pt")
    
    best_val_loss = float("inf")
    no_improve = 0
    
    if config.load_checkpoint and os.path.exists(ckpt_file):
        try:
            checkpoint = torch.load(ckpt_file, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                best_val_loss = checkpoint.get('best_val_loss', float("inf"))
                logger.log_msg(f"Loaded checkpoint with val_loss={best_val_loss:.3e}")
            else:
                model.load_state_dict(checkpoint)
                logger.log_msg("Loaded checkpoint (legacy format)")
        except Exception as e:
            logger.log_msg(f"Error loading checkpoint: {e}", level=30)
            logger.log_msg("Training from scratch")
    else:
        logger.log_msg("Training from scratch")

    # ─── Training loop ──────────────────────────────────────────────────
    try:
        for epoch in range(1, config.epochs + 1):
            # ----- Training phase ------------------------------------------
            model.train()
            train_loss_accum = 0.0
            train_batches = 0
            
            progress_bar = tqdm(dl_train, desc=f"Epoch {epoch}/{config.epochs} [Train]")
            for batch in progress_bar:
                # Unpack batch
                xb, yb = batch[0], batch[1]  # Returns and parameters
                
                # Move to device and normalize
                xb = _standardise(xb.to(device, non_blocking=True), x_mean, x_std)
                yb = yb.to(device, non_blocking=True)
                yb_norm = (yb - y_mean) / y_std
                
                # Forward pass
                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=config.mixed_precision):
                    preds_norm = model(xb, config.mu_mode)
                    
                    # Ensure dimensions match
                    if preds_norm.shape != yb_norm.shape:
                        raise RuntimeError(
                            f"Prediction shape {preds_norm.shape} doesn't match "
                            f"target shape {yb_norm.shape}"
                        )
                    
                    # Loss computation (heteroscedastic if log_vars available)
                    if hasattr(model, 'log_vars') and model.log_vars is not None:
                        log_vars = model.log_vars
                        if log_vars.numel() == effective_n_params:
                            mse = (preds_norm - yb_norm).pow(2)
                            loss = (torch.exp(-log_vars) * mse + log_vars).mean()
                        else:
                            # Fallback to standard MSE if dimensions don't match
                            loss = loss_fn(preds_norm, yb_norm).mean()
                    else:
                        loss = loss_fn(preds_norm, yb_norm).mean()
                
                # Backward pass
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                
                train_loss_accum += loss.item()
                train_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.3e}',
                    'lr': f'{opt.param_groups[0]["lr"]:.1e}'
                })
            
            avg_train_loss = train_loss_accum / train_batches
            logger.log(epoch, avg_train_loss, "train/loss")
            
            # ----- Validation phase ----------------------------------------
            if epoch % config.validate_every == 0:
                try:
                    r2_per_param, r2_global, avg_val_loss = compute_r2_incremental(
                        model, dl_val, device, x_mean, x_std, y_mean, y_std, config
                    )
                    
                    # Logging
                    logger.log(epoch, avg_val_loss, "val/loss")
                    logger.log(epoch, r2_global, "val/R2_global")
                    
                    for i, r2p in enumerate(r2_per_param):
                        logger.log(epoch, r2p.item(), f"val/R2_param_{i}")
                        if i < len(param_order):
                            param_name = param_order[i] if config.mu_mode == "raw" else param_order[i+1]
                            logger.log(epoch, r2p.item(), f"val/R2_{param_name}")
                    
                    logger.log_msg(
                        f"[{epoch:03d}] train_loss={avg_train_loss:.3e} "
                        f"val_loss={avg_val_loss:.3e} R²_global={r2_global:.4f}"
                    )
                    
                    # Scheduler and early stopping
                    sched.step(avg_val_loss)
                    
                    if avg_val_loss < best_val_loss - 1e-8:
                        best_val_loss = avg_val_loss
                        no_improve = 0
                        
                        # Save comprehensive checkpoint
                        checkpoint = {
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': opt.state_dict(),
                            'scheduler_state_dict': sched.state_dict(),
                            'best_val_loss': best_val_loss,
                            'epoch': epoch,
                            'r2_global': r2_global,
                            'r2_per_param': r2_per_param.cpu().numpy(),
                            'config': config.__dict__,
                            'normalization': {
                                'x_mean': x_mean.cpu(), 'x_std': x_std.cpu(),
                                'y_mean': y_mean.cpu(), 'y_std': y_std.cpu()
                            },
                            'param_order': param_order,
                        }
                        torch.save(checkpoint, ckpt_file)
                        logger.log_msg("✓ Improved - checkpoint saved")
                    else:
                        no_improve += 1
                        logger.log_msg(f"No improvement ({no_improve}/{config.early_stop_patience})")
                        
                        if no_improve >= config.early_stop_patience:
                            logger.log_msg("Early stopping triggered", level=40)
                            break
                            
                except Exception as e:
                    logger.log_msg(f"Error during validation: {e}", level=40)
                    # Continue training even if validation fails
                    continue
            
            else:
                logger.log_msg(f"[{epoch:03d}] train_loss={avg_train_loss:.3e}")

    except KeyboardInterrupt:
        logger.log_msg("Training interrupted by user", level=30)
    except Exception as e:
        logger.log_msg(f"Training error: {e}", level=40)
        raise
    finally:
        # ─── Restore best model ────────────────────────────────────────
        if os.path.exists(ckpt_file):
            try:
                checkpoint = torch.load(ckpt_file, map_location=device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    final_r2 = checkpoint.get('r2_global', 0.0)
                    final_epoch = checkpoint.get('epoch', '?')
                    logger.log_msg(f"Restored best model (epoch {final_epoch}, R²={final_r2:.4f})")
                else:
                    model.load_state_dict(checkpoint)
                    logger.log_msg("Restored best model")
            except Exception as e:
                logger.log_msg(f"Error loading best checkpoint: {e}", level=30)
        
        logger.close()

    return model


# ───────────────────────────────────────────────────────────────────────────
# Model evaluation utilities
# ---------------------------------------------------------------------------
@torch.inference_mode()
def validate_model(
    model: nn.Module, 
    dataset: 'SVFamilyDataset',
    config: 'TrainingConfig',
    device: str = "cuda",
    batch_size: int = 512,
    checkpoint_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Comprehensive model validation with detailed metrics.
    
    Parameters
    ----------
    model : nn.Module
        Trained model to evaluate.
    dataset : SVFamilyDataset
        Dataset to evaluate on.
    config : TrainingConfig
        Training configuration.
    device : str, default="cuda"
        Device for evaluation.
    batch_size : int, default=512
        Batch size for evaluation.
    checkpoint_path : str, optional
        Path to checkpoint with normalization stats.
    
    Returns
    -------
    dict
        Validation results and metrics.
    """
    model.eval()
    device = torch.device(device)
    model = model.to(device)
    
    # Load normalization stats if available
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'normalization' in checkpoint:
            norm_stats = checkpoint['normalization']
            x_mean = norm_stats['x_mean'].to(device)
            x_std = norm_stats['x_std'].to(device)
            y_mean = norm_stats['y_mean'].to(device)
            y_std = norm_stats['y_std'].to(device)
        else:
            raise ValueError("Checkpoint missing normalization statistics")
    else:
        # Compute normalization on the fly
        dl_norm = loader(dataset, batch_size=min(2048, len(dataset)), 
                        shuffle=False, num_workers=0)
        x_mean, x_std = estimate_input_norm(dl_norm, device)
        y_mean, y_std = compute_target_statistics(dl_norm, device, config.mu_mode)
    
    # Evaluation dataloader
    dl = loader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    all_predictions = []
    all_targets = []
    all_losses = []
    
    with torch.no_grad():
        for batch in tqdm(dl, desc="Evaluating"):
            xb, yb = batch[0], batch[1]
            xb = _standardise(xb.to(device), x_mean, x_std)
            yb = yb.to(device)
            
            # Get predictions
            preds_norm = model(xb, config.mu_mode)
            
            # Denormalize predictions
            preds = preds_norm * y_std + y_mean
            
            # Clip predictions if requested
            if config.clip_predictions:
                param_order = get_param_order(config.model_type)
                if config.mu_mode == "analytical":
                    param_order = param_order[1:]
                preds = clip_sv_parameters(preds, config.model_type, param_order)
            
            all_predictions.append(preds.cpu())
            all_targets.append(yb.cpu())
            
            # Loss in normalized space
            yb_norm = (yb - y_mean) / y_std
            loss = nn.MSELoss(reduction='none')(preds_norm, yb_norm)
            all_losses.append(loss.cpu())
    
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    losses = torch.cat(all_losses, dim=0)
    
    # Compute comprehensive metrics
    mse = ((predictions - targets) ** 2).mean(dim=0)
    mae = (predictions - targets).abs().mean(dim=0)
    mape = (((predictions - targets).abs() / (targets.abs() + 1e-8)) * 100).mean(dim=0)
    
    # R² per parameter
    ss_res = ((predictions - targets) ** 2).sum(dim=0)
    ss_tot = ((targets - targets.mean(dim=0)) ** 2).sum(dim=0)
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    
    # Parameter-specific statistics
    param_order = get_param_order(config.model_type)
    if config.mu_mode == "analytical":
        param_order = param_order[1:]
    
    param_metrics = {}
    for i, param_name in enumerate(param_order):
        param_metrics[param_name] = {
            'mse': float(mse[i]),
            'mae': float(mae[i]),
            'mape': float(mape[i]),
            'r2': float(r2[i]),
            'target_mean': float(targets[:, i].mean()),
            'target_std': float(targets[:, i].std()),
            'pred_mean': float(predictions[:, i].mean()),
            'pred_std': float(predictions[:, i].std()),
        }
    
    return {
        'global_metrics': {
            'mse_global': float(mse.mean()),
            'mae_global': float(mae.mean()),
            'mape_global': float(mape.mean()),
            'r2_global': float(r2.mean()),
            'loss_normalized': float(losses.mean()),
        },
        'parameter_metrics': param_metrics,
        'predictions': predictions.numpy(),
        'targets': targets.numpy(),
        'config': config.__dict__,
        'normalization_stats': {
            'x_mean': x_mean.cpu().numpy(),
            'x_std': x_std.cpu().numpy(),
            'y_mean': y_mean.cpu().numpy(),
            'y_std': y_std.cpu().numpy(),
        }
    }


# ───────────────────────────────────────────────────────────────────────────
# Convenience functions for backward compatibility
# ---------------------------------------------------------------------------
def train_sv_model(
    arch: str = "cnn",
    model_type: str = "svcj", 
    seq_len: int = 672,
    epochs: int = 100,
    batch_size: int = 512,
    n_samples: int = 110_000,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    logdir: str = "runs/sv_family",
    device: Optional[str] = None,
    val_split: float = 0.1,
    early_stop_patience: int = 6,
    num_workers: int = 4,
    mixed_precision: bool = True,
    mu_mode: str = "raw",
    load_checkpoint: bool = False,
    seed: int = 42,
    **kwargs
) -> nn.Module:
    """
    Backward-compatible training function that matches the original signature.
    """
    config = TrainingConfig(
        arch=arch, model_type=model_type, seq_len=seq_len, epochs=epochs,
        batch_size=batch_size, n_samples=n_samples, lr=lr, weight_decay=weight_decay,
        logdir=logdir, device=device, val_split=val_split, 
        early_stop_patience=early_stop_patience, num_workers=num_workers,
        mixed_precision=mixed_precision, mu_mode=mu_mode, 
        load_checkpoint=load_checkpoint, seed=seed, **kwargs
    )
    
    return train(config)


if __name__ == "__main__":
    # Example: Proper configuration for preserving drift signal (μ estimation)
    config = TrainingConfig(
        arch="cnn_ficura",
        model_type="svj",
        dt=1.0/365  ,              # 1/252 - auto-converts annualized ranges
        seq_len=2000,               # Their sequence length
        n_samples=10_000,           # Their training size
        batch_size=512,             # Their batch size
        mu_mode="raw",              # Predict all parameters including μ
        
        # CRITICAL: Preserve drift signal for μ estimation
        normalize_returns=False,    # No per-series centering in dataset
        standardize_params=False,   # No per-param norm in dataset
        # → Global normalization will be applied in training to preserve 
        #   cross-sample μ differences while normalizing scale
    )

    model = train(config)  # Drift signal preserved for μ estimation!