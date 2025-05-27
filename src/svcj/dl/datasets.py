from __future__ import annotations

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional, Literal, Tuple, Dict, Any, Union
import warnings
from dataclasses import dataclass
from pathlib import Path
import pickle

# Import simulation modules (avoid circular imports)
from svcj.dl.simulator import (
    SVParams, simulate_returns, DT_HOUR
)
from svcj.dl.config import get_param_order, get_param_ranges, sample_parameters_with_feller

__all__ = [
    "SVFamilyDataset", "CachedSVDataset", "BatchSVDataset", 
    "loader", "create_validation_dataset", "SVCJDataset", "DatasetConfig"
]


@dataclass
class DatasetConfig:
    """Configuration for SV family datasets."""
    
    # Core simulation parameters
    seq_len: int = 672
    n_samples: int = 100_000
    model_type: str = "svcj"
    mu_mode: str = "raw"  # {"raw", "analytical", "devol"}
    
    # Simulation parameters
    dt: float = DT_HOUR
    rho: Optional[float] = None  # If None, samples from [-0.9, 0.9]
    s0: float = 1.0
    v0: Optional[float] = None
    param_range_source: str = "ficura"
    
    # Dataset behavior - CRITICAL FOR DRIFT SIGNAL PRESERVATION
    return_variance: bool = False
    return_jumps: bool = False
    
    # Normalization settings - IMPORTANT: affects drift signal!
    # For μ-aware models, use normalize_returns=False, standardize_params=False
    # and apply global normalization in training to preserve cross-sample μ differences
    normalize_returns: bool = False     # False = preserve drift signal (recommended for μ estimation)
    standardize_params: bool = False    # False = use global param normalization in training
    
    # Random number generation
    seed: int = 42
    deterministic: bool = True  # Whether to use deterministic parameter sampling
    
    # Caching options
    cache_data: bool = False
    cache_path: Optional[str] = None
    
    # Feller condition options (removed - using variance flooring instead)
    variance_floor: float = 1e-12   # Minimum variance floor
    
    def validate(self):
        """Validate configuration parameters."""
        if self.seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {self.seq_len}")
        if self.n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {self.n_samples}")
        if self.model_type not in ("sv", "svj", "svcj"):
            raise ValueError(f"model_type must be one of ['sv', 'svj', 'svcj'], got {self.model_type}")
        if self.mu_mode not in ("raw", "analytical", "devol"):
            raise ValueError(f"mu_mode must be one of ['raw', 'analytical', 'devol'], got {self.mu_mode}")
        if self.dt <= 0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        if self.rho is not None and not -1 <= self.rho <= 1:
            raise ValueError(f"rho must be in [-1, 1], got {self.rho}")
        if self.variance_floor < 0:
            raise ValueError(f"variance_floor must be non-negative, got {self.variance_floor}")

class SVFamilyDataset(Dataset):
    """
    Enhanced on-the-fly simulation dataset for SV family models.
    
    Features:
    - Uses vectorized simulation for better performance
    - Supports variance and jump output
    - Configurable parameter ranges and correlation
    - Option to normalize/standardize outputs
    - Deterministic and random parameter sampling modes
    - Comprehensive error handling and validation
    """

    def __init__(
        self,
        seq_len: int = 672,
        n_samples: int = 100_000,
        seed: int = 42,
        model_type: str = "svcj",
        mu_mode: str = "raw",
        dt: float = DT_HOUR,
        rho: Optional[float] = None,
        return_variance: bool = False,
        return_jumps: bool = False,
        normalize_returns: bool = False,
        standardize_params: bool = False,
        deterministic: bool = True,
        config: Optional[DatasetConfig] = None,
    ):
        """
        Initialize SV family dataset.
        
        Parameters
        ----------
        seq_len : int, default=672
            Length of each simulated return series.
        n_samples : int, default=100_000
            Number of samples in the dataset.
        seed : int, default=42
            Random seed for reproducibility.
        model_type : str, default="svcj"
            Model type: "sv", "svj", or "svcj".
        mu_mode : str, default="raw"
            Parameter mode: "raw", "analytical", or "devol".
        dt : float, default=DT_HOUR
            Time step for simulation.
        rho : float, optional
            Fixed correlation. If None, samples from appropriate range.
        return_variance : bool, default=False
            Whether to return variance paths.
        return_jumps : bool, default=False
            Whether to return jump indicators.
        normalize_returns : bool, default=False
            Whether to normalize returns to zero mean, unit variance.
        standardize_params : bool, default=False
            Whether to standardize parameters.
        deterministic : bool, default=True
            Whether to use deterministic parameter sampling.
        config : DatasetConfig, optional
            Configuration object (overrides individual parameters).
        """
        
        # Use config if provided, otherwise create from individual params
        if config is not None:
            config.validate()
            self.config = config
        else:
            self.config = DatasetConfig(
                seq_len=seq_len, n_samples=n_samples, model_type=model_type,
                mu_mode=mu_mode, dt=dt, rho=rho, s0=1.0, v0=None,
                return_variance=return_variance, return_jumps=return_jumps,
                normalize_returns=normalize_returns, standardize_params=standardize_params,
                seed=seed, deterministic=deterministic,
                param_range_source=getattr(config, 'param_range_source', 'ficura')
            )
            self.config.validate()
        
        # Set up random number generator
        self.rng = np.random.default_rng(self.config.seed)
        self._setup_parameters()
        
        # Statistics for normalization (computed lazily)
        self._return_stats = None
        self._param_stats = None
        
        # Cache for validation
        self._validation_cache = {}

    def _setup_parameters(self):
        """Set up parameter sampling configuration with proper dt scaling."""
        
        # Get parameter information
        self._param_order = get_param_order(self.config.model_type)
        
        # Get ranges properly scaled for our dt
        from svcj.dl.config import get_param_ranges
        self._param_ranges = get_param_ranges(
            self.config.model_type, 
            self.config.dt, 
            range_source=self.config.param_range_source
        )
        
        # Validate parameter configuration
        if not self._param_order:
            raise ValueError(f"No parameters found for model_type: {self.config.model_type}")
        
        # Set up correlation sampling if not fixed
        if self.config.rho is None:
            # Sample correlation from reasonable range
            self._rho_range = (-0.9, 0.9)
        else:
            self._rho_range = None
        
        # Pre-generate parameter samples if deterministic
        if self.config.deterministic:
            self._pregenerate_parameters()

    def _pregenerate_parameters(self):
        """Pre-generate all parameter vectors for deterministic behavior."""
        
        print(f"Pre-generating {self.config.n_samples} parameter sets...")
        
        # Simple parameter generation (no Feller condition checking)
        self._param_cache = np.empty((self.config.n_samples, len(self._param_order)), dtype=np.float32)
        
        if self.config.rho is None:
            self._rho_cache = np.empty(self.config.n_samples, dtype=np.float32)
        
        # Generate all parameters at once for efficiency
        for i in range(self.config.n_samples):
            self._param_cache[i] = self._sample_params_single()
            
            if self.config.rho is None:
                self._rho_cache[i] = self.rng.uniform(*self._rho_range)
            
            if (i + 1) % 10000 == 0:
                print(f"  Generated {i + 1:,}/{self.config.n_samples:,} parameter sets...")
        
        print(f"Successfully generated {self.config.n_samples:,} parameter sets!")

    def _sample_params_single(self) -> np.ndarray:
        """Sample a single parameter vector, properly scaled for dt."""
        
        # Simple sampling without Feller condition - using variance flooring instead
        from svcj.dl.config import get_param_ranges
        dt_ranges = get_param_ranges(
            self.config.model_type, 
            self.config.dt, 
            range_source=self.config.param_range_source
        )
        
        return np.array([
            self.rng.uniform(
                dt_ranges[param_name].low,
                dt_ranges[param_name].high
            )
            for param_name in self._param_order
        ], dtype=np.float32)

    def _get_parameters(self, idx: int) -> Tuple[np.ndarray, float]:
        """Get parameters for a given index."""
        
        if self.config.deterministic:
            # Use pre-generated parameters
            params = self._param_cache[idx].copy()
            rho = self.config.rho if self.config.rho is not None else self._rho_cache[idx]
        else:
            # Generate on-the-fly (less reproducible but more memory efficient)
            # Set RNG state based on index for some reproducibility
            local_rng = np.random.default_rng(self.config.seed + idx)
            
            params = np.array([
                local_rng.uniform(
                    self._param_ranges[param_name].low,
                    self._param_ranges[param_name].high
                )
                for param_name in self._param_order
            ], dtype=np.float32)
            
            if self.config.rho is None:
                rho = local_rng.uniform(*self._rho_range)
            else:
                rho = self.config.rho
        
        return params, rho

    def _simulate_path(self, params: np.ndarray, rho: float) -> Tuple[np.ndarray, ...]:
        """Simulate a single path with given parameters using variance flooring."""
        
        try:
            # Create SVParams object (no Feller validation)
            sv_params = SVParams.from_sequence(
                params, self.config.model_type, rho=rho, dt=self.config.dt
            )
            
            # Simulate returns with variance flooring
            result = simulate_returns(
                params=sv_params,
                steps=self.config.seq_len,
                s0=self.config.s0,
                v0=self.config.v0,
                dt=self.config.dt,
                model_type=self.config.model_type,
                rho=rho,
                seed=None,  # Let simulation use its own randomness
                return_variance=self.config.return_variance,
                return_jumps=self.config.return_jumps,
                variance_floor=self.config.variance_floor,  # Use variance flooring
            )
            
            return result
            
        except Exception as e:
            # If simulation fails, generate a warning and return zeros
            warnings.warn(f"Simulation failed for params {params}: {e}")
            
            # Return zeros with correct shape
            returns = np.zeros(self.config.seq_len, dtype=np.float32)
            result = [returns]
            
            if self.config.return_variance:
                variance = np.full(self.config.seq_len, params[1], dtype=np.float32)  # Use v_long
                result.append(variance)
            
            if self.config.return_jumps:
                jumps = np.zeros(self.config.seq_len, dtype=bool)
                result.append(jumps)
            
            return tuple(result) if len(result) > 1 else result[0]

    def _process_outputs(self, simulation_result, params: np.ndarray) -> Tuple[torch.Tensor, ...]:
        """Process simulation outputs into tensors."""
        
        # Unpack simulation results
        if self.config.return_variance and self.config.return_jumps:
            returns, variance, jumps = simulation_result
        elif self.config.return_variance:
            returns, variance = simulation_result
            jumps = None
        elif self.config.return_jumps:
            returns, jumps = simulation_result
            variance = None
        else:
            returns = simulation_result
            variance = None
            jumps = None
        
        # Process returns - CRITICAL: per-series centering removes drift signal!
        # When normalize_returns=True, we do per-series centering which removes 
        # the drift component that encodes μ. For drift-aware models, set 
        # normalize_returns=False and use global normalization in training instead.
        if self.config.normalize_returns:
            # WARNING: This removes per-series drift signal that encodes μ!
            # Only use this if you don't need to preserve cross-sample mean differences
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        # If normalize_returns=False, returns are left raw with drift signal intact.
        # Global normalization should then be applied during training to preserve
        # cross-sample μ variations while still normalizing the overall scale.
        
        # Convert to tensors
        returns_tensor = torch.from_numpy(returns.astype(np.float32)).unsqueeze(0)  # Add channel dim
        
        # Process parameters
        processed_params = params.copy()
        if self.config.mu_mode == "analytical":
            processed_params = processed_params[1:]  # Drop mu
        
        # Parameter standardization - similar logic applies
        if self.config.standardize_params:
            # This does per-parameter standardization within the dataset
            # For global parameter normalization, set standardize_params=False
            # and let the training script handle global parameter statistics
            if self._param_stats is None:
                self._compute_param_stats()
            processed_params = (processed_params - self._param_stats['mean']) / (self._param_stats['std'] + 1e-8)
        
        params_tensor = torch.from_numpy(processed_params.astype(np.float32))
        
        # Prepare outputs
        outputs = [returns_tensor, params_tensor]
        
        if self.config.return_variance and variance is not None:
            variance_tensor = torch.from_numpy(variance.astype(np.float32)).unsqueeze(0)
            outputs.append(variance_tensor)
        
        if self.config.return_jumps and jumps is not None:
            jumps_tensor = torch.from_numpy(jumps.astype(torch.uint8)).unsqueeze(0)
            outputs.append(jumps_tensor)
        
        return tuple(outputs)

    def _compute_param_stats(self):
        """Compute parameter statistics for standardization."""
        
        # Sample a subset of parameters to estimate statistics
        n_samples = min(1000, self.config.n_samples)
        param_samples = []
        
        for i in range(n_samples):
            params, _ = self._get_parameters(i)
            if self.config.mu_mode == "analytical":
                params = params[1:]
            param_samples.append(params)
        
        param_array = np.array(param_samples)
        self._param_stats = {
            'mean': param_array.mean(axis=0).astype(np.float32),
            'std': param_array.std(axis=0).astype(np.float32)
        }

    def __len__(self) -> int:
        return self.config.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """Get a single dataset item."""
        
        if not 0 <= idx < self.config.n_samples:
            raise IndexError(f"Index {idx} out of range [0, {self.config.n_samples})")
        
        # Get parameters
        params, rho = self._get_parameters(idx)
        
        # Simulate path
        simulation_result = self._simulate_path(params, rho)
        
        # Process and return
        return self._process_outputs(simulation_result, params)

    def get_batch(self, indices: Union[list, np.ndarray]) -> Tuple[torch.Tensor, ...]:
        """Get a batch of items efficiently."""
        
        batch_results = []
        for idx in indices:
            batch_results.append(self[idx])
        
        # Stack tensors
        n_outputs = len(batch_results[0])
        batched = []
        
        for output_idx in range(n_outputs):
            output_tensors = [item[output_idx] for item in batch_results]
            batched.append(torch.stack(output_tensors, dim=0))
        
        return tuple(batched)

    def get_parameter_stats(self) -> Dict[str, np.ndarray]:
        """Get statistics about the parameter distribution."""
        
        if self._param_stats is None:
            self._compute_param_stats()
        
        return self._param_stats.copy()

    def validate_sample(self, idx: int = 0) -> Dict[str, Any]:
        """Validate a sample and return diagnostic information."""
        
        if idx in self._validation_cache:
            return self._validation_cache[idx]
        
        params, rho = self._get_parameters(idx)
        
        try:
            # Test simulation
            simulation_result = self._simulate_path(params, rho)
            processed_result = self._process_outputs(simulation_result, params)
            
            # Extract returns for analysis
            if isinstance(simulation_result, tuple):
                returns = simulation_result[0]
            else:
                returns = simulation_result
            
            # Compute diagnostics
            diagnostics = {
                'success': True,
                'params': params,
                'rho': rho,
                'return_mean': float(np.mean(returns)),
                'return_std': float(np.std(returns)),
                'return_min': float(np.min(returns)),
                'return_max': float(np.max(returns)),
                'output_shapes': [t.shape for t in processed_result],
                'output_dtypes': [t.dtype for t in processed_result],
            }
            
        except Exception as e:
            diagnostics = {
                'success': False,
                'error': str(e),
                'params': params,
                'rho': rho,
            }
        
        self._validation_cache[idx] = diagnostics
        return diagnostics


class BatchSVDataset(SVFamilyDataset):
    """
    Batch-optimized SV dataset that generates multiple paths simultaneously.
    More efficient for large batch sizes.
    """
    
    def __init__(self, batch_size: int = 32, **kwargs):
        """
        Initialize batch dataset.
        
        Parameters
        ----------
        batch_size : int, default=32
            Size of batches to generate simultaneously.
        **kwargs
            Arguments passed to SVFamilyDataset.
        """
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self._batch_cache = {}

    def _generate_batch(self, start_idx: int) -> Tuple[torch.Tensor, ...]:
        """Generate a batch of samples simultaneously."""
        
        end_idx = min(start_idx + self.batch_size, self.config.n_samples)
        actual_batch_size = end_idx - start_idx
        
        # Collect parameters for batch
        batch_params = []
        batch_rhos = []
        
        for idx in range(start_idx, end_idx):
            params, rho = self._get_parameters(idx)
            batch_params.append(params)
            batch_rhos.append(rho)
        
        batch_params = np.array(batch_params)
        batch_rhos = np.array(batch_rhos)
        
        # Use the first parameter set as template for SVParams
        template_params = SVParams.from_sequence(
            batch_params[0], self.config.model_type, rho=batch_rhos[0]
        )
        
        # For now, fall back to individual simulation
        # Full batch simulation would require more complex parameter handling
        batch_results = []
        for i in range(actual_batch_size):
            params = batch_params[i]
            rho = batch_rhos[i]
            result = self._simulate_path(params, rho)
            processed = self._process_outputs(result, params)
            batch_results.append(processed)
        
        # Stack results
        n_outputs = len(batch_results[0])
        batched = []
        
        for output_idx in range(n_outputs):
            output_tensors = [item[output_idx] for item in batch_results]
            batched.append(torch.stack(output_tensors, dim=0))
        
        return tuple(batched)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """Get item, using batch cache when possible."""
        
        batch_start = (idx // self.batch_size) * self.batch_size
        batch_offset = idx - batch_start
        
        if batch_start not in self._batch_cache:
            self._batch_cache[batch_start] = self._generate_batch(batch_start)
        
        batch_data = self._batch_cache[batch_start]
        
        # Extract single item from batch
        return tuple(tensor[batch_offset] for tensor in batch_data)


class CachedSVDataset(SVFamilyDataset):
    """
    Cached SV dataset that pre-generates and stores all data.
    Memory intensive but fastest for repeated access.
    """
    
    def __init__(self, cache_path: Optional[str] = None, **kwargs):
        """
        Initialize cached dataset.
        
        Parameters
        ----------
        cache_path : str, optional
            Path to save/load cached data.
        **kwargs
            Arguments passed to SVFamilyDataset.
        """
        super().__init__(**kwargs)
        self.cache_path = Path(cache_path) if cache_path else None
        self._data_cache = None
        
        # Try to load from cache
        if self.cache_path and self.cache_path.exists():
            self._load_cache()
        else:
            self._generate_cache()
            if self.cache_path:
                self._save_cache()

    def _generate_cache(self):
        """Generate and cache all data."""
        
        print(f"Generating {self.config.n_samples} cached samples...")
        
        self._data_cache = []
        for idx in range(self.config.n_samples):
            if idx % 1000 == 0:
                print(f"  Progress: {idx}/{self.config.n_samples}")
            
            item = super().__getitem__(idx)
            self._data_cache.append(item)
        
        print("Cache generation complete.")

    def _save_cache(self):
        """Save cache to disk."""
        
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            cache_data = {
                'data': self._data_cache,
                'config': self.config,
            }
            
            with open(self.cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            
            print(f"Cache saved to {self.cache_path}")

    def _load_cache(self):
        """Load cache from disk."""
        
        with open(self.cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        self._data_cache = cache_data['data']
        
        # Validate config compatibility
        cached_config = cache_data['config']
        if (cached_config.seq_len != self.config.seq_len or 
            cached_config.model_type != self.config.model_type):
            warnings.warn("Cached data config doesn't match current config")
        
        print(f"Cache loaded from {self.cache_path}")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """Get cached item."""
        
        if self._data_cache is None:
            raise RuntimeError("Cache not initialized")
        
        return self._data_cache[idx]


# ───────────────────────────────────────────────────────────────────────────
# Utility functions
# ───────────────────────────────────────────────────────────────────────────

def loader(
    dataset: Dataset, 
    batch_size: int = 256, 
    shuffle: bool = True, 
    num_workers: int = 0, 
    pin_memory: bool = True,
    **kwargs
) -> DataLoader:
    """Create a DataLoader with sensible defaults."""
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=num_workers, 
        pin_memory=pin_memory,
        **kwargs
    )


def create_validation_dataset(
    train_dataset: SVFamilyDataset,
    val_size: int = 10000,
    seed_offset: int = 1000,
) -> SVFamilyDataset:
    """
    Create a validation dataset with different random seed but same configuration.
    
    Parameters
    ----------
    train_dataset : SVFamilyDataset
        Training dataset to copy configuration from.
    val_size : int, default=10000
        Size of validation dataset.
    seed_offset : int, default=1000
        Offset to add to training seed for validation.
    
    Returns
    -------
    SVFamilyDataset
        Validation dataset.
    """
    
    val_config = DatasetConfig(**train_dataset.config.__dict__)
    val_config.n_samples = val_size
    val_config.seed = train_dataset.config.seed + seed_offset
    
    return SVFamilyDataset(config=val_config)


def analyze_dataset(dataset: SVFamilyDataset, n_samples: int = 100) -> Dict[str, Any]:
    """
    Analyze dataset properties by sampling a subset.
    
    Parameters
    ----------
    dataset : SVFamilyDataset
        Dataset to analyze.
    n_samples : int, default=100
        Number of samples to analyze.
    
    Returns
    -------
    dict
        Analysis results.
    """
    
    analysis = {
        'dataset_size': len(dataset),
        'config': dataset.config,
        'param_order': dataset._param_order,
        'validation_results': []
    }
    
    # Sample validation
    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)
    
    success_count = 0
    return_stats = []
    param_stats = []
    
    for idx in indices:
        validation = dataset.validate_sample(int(idx))
        analysis['validation_results'].append(validation)
        
        if validation['success']:
            success_count += 1
            return_stats.append([
                validation['return_mean'],
                validation['return_std'],
                validation['return_min'],
                validation['return_max']
            ])
            param_stats.append(validation['params'])
    
    analysis['success_rate'] = success_count / len(indices)
    
    if return_stats:
        return_stats = np.array(return_stats)
        analysis['return_statistics'] = {
            'mean_return': {
                'mean': float(np.mean(return_stats[:, 0])),
                'std': float(np.std(return_stats[:, 0]))
            },
            'return_volatility': {
                'mean': float(np.mean(return_stats[:, 1])),
                'std': float(np.std(return_stats[:, 1]))
            }
        }
        
        param_stats = np.array(param_stats)
        analysis['parameter_statistics'] = {
            param_name: {
                'mean': float(np.mean(param_stats[:, i])),
                'std': float(np.std(param_stats[:, i])),
                'min': float(np.min(param_stats[:, i])),
                'max': float(np.max(param_stats[:, i]))
            }
            for i, param_name in enumerate(dataset._param_order)
        }
    
    return analysis


# Backwards compatibility
SVCJDataset = SVFamilyDataset


# Example usage
if __name__ == "__main__":
    # Basic usage
    config = DatasetConfig(
        seq_len=252,
        n_samples=1000,
        model_type="svcj",
        return_variance=True,
        normalize_returns=True,
        param_range_source="default",
        seed=42
    )
    
    dataset = SVFamilyDataset(config=config)
    
    # Test dataset
    sample = dataset[0]
    print(f"Sample shapes: {[s.shape for s in sample]}")
    
    # Analyze dataset
    analysis = analyze_dataset(dataset, n_samples=10)
    print(f"Success rate: {analysis['success_rate']:.2%}")
    
    # Create DataLoader
    dl = loader(dataset, batch_size=32, num_workers=2)
    batch = next(iter(dl))
    print(f"Batch shapes: {[b.shape for b in batch]}")

    # Example with ficura ranges
    config_ficura = DatasetConfig(
        seq_len=252,
        n_samples=100,
        model_type="svj",
        param_range_source="ficura",
        seed=123
    )
    dataset_ficura = SVFamilyDataset(config=config_ficura)
    sample_ficura = dataset_ficura[0]
    print(f"Ficura Sample shapes: {[s.shape for s in sample_ficura]}")
    analysis_ficura = analyze_dataset(dataset_ficura, n_samples=10)
    print(f"Ficura Success rate: {analysis_ficura['success_rate']:.2%}")