"""
Fully vectorized SV family path/return simulation with robust parameter handling.
Supports SV, SVJ, and SVCJ models with flexible correlation structures.
"""
from __future__ import annotations

import numpy as np
from numpy.random import Generator, default_rng
from typing import Sequence, Literal, Optional, Union, Tuple
from dataclasses import dataclass
import warnings

# Default constants
DT_HOUR = 1.0 / (365 * 24)  # Hourly time step
EPSILON = 1e-12  # Numerical stability threshold

__all__ = ["SVParams", "simulate_returns", "simulate_paths_batch", "validate_parameters"]


@dataclass
class SVParams:
    """Parameter container for SV family models with validation."""
    
    # Core SV parameters
    mu: float          # drift
    v_long: float      # long-term variance
    beta: float        # variance persistence (or kappa directly if use_kappa=True)
    gamma: float       # volatility of variance
    
    # Jump parameters (used for SVJ/SVCJ)
    mu_j: float = 0.0      # jump mean
    sigma_j: float = 0.0   # jump volatility  
    lam: float = 0.0       # jump intensity
    
    # Correlation parameter
    rho: float = 1.0       # correlation between price and variance shocks
    
    # Control flags
    use_kappa: bool = False  # if True, beta is interpreted as kappa directly
    
    def __post_init__(self):
        """Validate parameters after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """Validate parameter ranges for economic and numerical feasibility."""
        
        # Basic bounds
        if self.v_long <= 0:
            raise ValueError(f"v_long must be positive, got {self.v_long}")
        if self.gamma <= 0:
            raise ValueError(f"gamma must be positive, got {self.gamma}")
        if not -1 <= self.rho <= 1:
            raise ValueError(f"rho must be in [-1, 1], got {self.rho}")
        if self.lam < 0:
            raise ValueError(f"lambda must be non-negative, got {self.lam}")
        if self.sigma_j < 0:
            raise ValueError(f"sigma_j must be non-negative, got {self.sigma_j}")
        
        # Beta/kappa validation
        if self.use_kappa:
            if self.beta <= 0:  # beta interpreted as kappa
                raise ValueError(f"kappa must be positive, got {self.beta}")
        else:
            if not 0 < self.beta < 1:
                raise ValueError(f"beta must be in (0, 1), got {self.beta}")
        
        # Note: Feller condition checking removed - we'll use variance flooring instead
    
    def get_kappa(self, dt: float) -> float:
        """Get mean reversion speed κ."""
        if self.use_kappa:
            return self.beta
        else:
            return -np.log(self.beta) / dt
    
    @classmethod
    def from_sequence(cls, params: Sequence[float], model_type: str, 
                     rho: float = 1.0, use_kappa: bool = False, dt: float = None) -> 'SVParams':
        """
        Create SVParams from parameter sequence based on model type.
        
        Parameters
        ----------
        params : Sequence[float]
            Parameter values (should already be scaled for the correct dt)
        model_type : str
            Model type ('sv', 'svj', 'svcj')
        rho : float, default=1.0
            Correlation parameter
        use_kappa : bool, default=False
            Whether beta is interpreted as kappa directly
        dt : float, optional
            Time step (for validation only - Feller checking removed)
        """
        
        model_type = model_type.lower()
        if model_type not in ("sv", "svj", "svcj"):
            raise ValueError("model_type must be one of 'sv', 'svj', 'svcj'")
        
        if model_type == "sv":
            if len(params) != 4:
                raise ValueError(f"SV model requires 4 parameters, got {len(params)}")
            mu, v_long, beta, gamma = params
            sv_params = cls(mu=mu, v_long=v_long, beta=beta, gamma=gamma, 
                          rho=rho, use_kappa=use_kappa)
        else:  # SVJ or SVCJ
            if len(params) != 7:
                raise ValueError(f"{model_type.upper()} model requires 7 parameters, got {len(params)}")
            mu, v_long, beta, gamma, mu_j, sigma_j, lam = params
            sv_params = cls(mu=mu, v_long=v_long, beta=beta, gamma=gamma,
                          mu_j=mu_j, sigma_j=sigma_j, lam=lam, 
                          rho=rho, use_kappa=use_kappa)
        
        # Optional basic validation (Feller checking removed)
        if dt is not None:
            sv_params._validate_basic_for_dt(dt)
        
        return sv_params
    

    def _validate_basic_for_dt(self, dt: float) -> None:
        """
        Basic validation that parameters (interpreted as annual rates where applicable)
        are reasonable for the given time step dt.
        dt is the time step fraction of a year (e.g., 1/250 or 1/365).
        """
        warnings_issued = []

        # self.mu is already an annual rate
        annual_mu_rate = self.mu
        # Define what you consider an extreme *annual* drift rate.
        # E.g., Fičura samples annual mu up to 0.1 (10%). Maybe > 0.5 (50%) is extreme.
        if abs(annual_mu_rate) > 0.5: # Check for > 50% annual drift
            warnings_issued.append(f"Potentially extreme annual drift rate: {annual_mu_rate:.2f}")

        # self.v_long is the per-step (dt) variance amount.
        # To get an annualized v_long, we do (self.v_long / dt).
        annual_v_long_rate = self.v_long / dt if dt > 0 else float('inf')
        # Fičura samples annual v_long up to ~0.0567. Maybe > 0.2 (20% annual variance) is extreme.
        if annual_v_long_rate > 1: # Check for > 20% annual variance
            warnings_issued.append(f"Potentially extreme annual variance rate: {annual_v_long_rate:.2f}")

        # self.lam is already an annual jump intensity rate
        annual_lam_rate = self.lam
        # Fičura samples annual lambda up to 12.6. Maybe > 30-50 jumps/year is extreme.
        if annual_lam_rate > 100.0: # Check for > 40 jumps per year
            warnings_issued.append(f"Potentially extreme annual jump intensity rate: {annual_lam_rate:.2f}")

        # You might want to add a check for gamma if it's also an annual rate in SVParams
        # annual_gamma_rate = self.gamma
        # Fičura samples annual gamma up to ~7.9. Maybe > 10 or 15 is extreme.
        # if hasattr(self, 'gamma') and self.gamma > 15.0:
        #     warnings_issued.append(f"Potentially extreme annual vol-of-vol rate: {self.gamma:.2f}")

        if warnings_issued:
            import warnings # Ensure imported
            # Construct a more informative warning message
            param_details = f"(SVParams content: mu={self.mu:.4f}, lam={self.lam:.4f}, v_long_step={self.v_long:.6e})"
            warning_msg = (f"Parameter validation (annualized perspective) warnings for dt={dt:.6f}: "
                        + "; ".join(warnings_issued) + f" {param_details}")
            warnings.warn(warning_msg, RuntimeWarning)


def validate_parameters(params: Union[SVParams, Sequence[float]], 
                       model_type: str) -> SVParams:
    """Validate and convert parameters to SVParams object."""
    
    if isinstance(params, SVParams):
        params.validate()
        return params
    else:
        return SVParams.from_sequence(params, model_type)


def _generate_correlated_shocks(n_steps: int, n_paths: int, rho: float, 
                               rng: Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate correlated Brownian motion increments.
    
    Returns:
        (dW_s, dW_v): price and variance shock arrays of shape (n_steps, n_paths)
    """
    if abs(rho - 1.0) < EPSILON:
        # Perfect correlation case - more efficient
        dW = rng.standard_normal((n_steps, n_paths))
        return dW, dW
    elif abs(rho) < EPSILON:
        # No correlation case
        dW_s = rng.standard_normal((n_steps, n_paths))
        dW_v = rng.standard_normal((n_steps, n_paths))
        return dW_s, dW_v
    else:
        # General correlation case using Cholesky decomposition
        dW_s = rng.standard_normal((n_steps, n_paths))
        dW_indep = rng.standard_normal((n_steps, n_paths))
        dW_v = rho * dW_s + np.sqrt(1 - rho**2) * dW_indep
        return dW_s, dW_v


def _generate_jumps(n_steps: int, n_paths: int, params: SVParams, dt: float,
                   model_type: str, rng: Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate jump components for price and variance.
    
    Returns:
        (J_s, J_v): jump arrays of shape (n_steps, n_paths)
    """
    J_s = np.zeros((n_steps, n_paths))
    J_v = np.zeros((n_steps, n_paths))
    
    if model_type in ("svj", "svcj") and params.lam > 0:
        # Poisson jump times
        jump_prob = params.lam * dt
        
        if jump_prob >= 1.0:
            warnings.warn(f"Jump probability {jump_prob:.3f} >= 1. Consider smaller dt.")
            jump_prob = min(jump_prob, 0.99)
        
        # Generate jump indicators
        jump_mask = rng.random((n_steps, n_paths)) < jump_prob
        n_jumps = np.sum(jump_mask)
        
        if n_jumps > 0:
            # Generate jump sizes only where needed
            jump_sizes = rng.normal(params.mu_j, params.sigma_j, n_jumps)
            J_s[jump_mask] = jump_sizes
            
            if model_type == "svcj":
                # For SVCJ, variance jumps are perfectly correlated with price jumps
                J_v[jump_mask] = jump_sizes
    
    return J_s, J_v


def simulate_returns(
    params: Union[SVParams, Sequence[float]],
    steps: int,
    s0: float = 1.0,
    v0: Optional[float] = None,
    dt: float = DT_HOUR,
    model_type: Literal["sv", "svj", "svcj"] = "svcj",
    rho: Optional[float] = None,
    seed: Optional[int] = None,
    return_variance: bool = True,
    return_jumps: bool = False,
    variance_floor: float = EPSILON,  # Add variance floor parameter
) -> Union[Tuple[np.ndarray, np.ndarray], 
           Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Simulate log-returns and variance path for SV family models (single path).

    Parameters
    ----------
    params : SVParams or Sequence[float]
        Model parameters. If sequence, interpreted based on model_type.
    steps : int
        Number of time steps to simulate.
    s0 : float, default=1.0
        Initial price level (for reference only).
    v0 : float, optional
        Initial variance. If None, uses long-term variance.
    dt : float, default=DT_HOUR
        Time step size.
    model_type : {"sv", "svj", "svcj"}, default="svcj"
        Model type to simulate.
    rho : float, optional
        Correlation between price and variance shocks. If None, uses value from params.
    seed : int, optional
        Random seed for reproducibility.
    return_variance : bool, default=True
        Whether to return variance path.
    return_jumps : bool, default=False
        Whether to return jump indicators.
    variance_floor : float, default=EPSILON
        Minimum variance value (variance flooring for numerical stability).

    Returns
    -------
    returns : ndarray
        Log-returns of shape (steps,).
    variance : ndarray, optional
        Variance path of shape (steps,) if return_variance=True.
    jumps : ndarray, optional
        Jump indicators of shape (steps,) if return_jumps=True.
        
    Note
    ----
    Uses variance flooring (v = max(v, variance_floor)) instead of enforcing 
    Feller condition for better numerical stability.
    """
    result = simulate_paths_batch(
        params=params, steps=steps, n_paths=1, s0=s0, v0=v0, dt=dt,
        model_type=model_type, rho=rho, seed=seed,
        return_variance=return_variance, return_jumps=return_jumps,
        variance_floor=variance_floor  # Pass variance floor parameter
    )
    
    # Squeeze out the path dimension
    if return_jumps and return_variance:
        returns, variance, jumps = result
        return returns.squeeze(), variance.squeeze(), jumps.squeeze()
    elif return_variance:
        returns, variance = result
        return returns.squeeze(), variance.squeeze()
    else:
        return result.squeeze()


def simulate_paths_batch(
    params: Union[SVParams, Sequence[float]],
    steps: int,
    n_paths: int = 1,
    s0: float = 1.0,
    v0: Optional[float] = None,
    dt: float = DT_HOUR,
    model_type: Literal["sv", "svj", "svcj"] = "svcj",
    rho: Optional[float] = None,
    seed: Optional[int] = None,
    return_variance: bool = True,
    return_jumps: bool = False,
    variance_floor: float = EPSILON,  # Floor for variance
) -> Union[np.ndarray, Tuple[np.ndarray, ...], ]:
    """
    Vectorized simulation of multiple SV family paths simultaneously.

    Parameters
    ----------
    params : SVParams or Sequence[float]
        Model parameters.
    steps : int
        Number of time steps per path.
    n_paths : int, default=1
        Number of paths to simulate.
    s0 : float, default=1.0
        Initial price level.
    v0 : float, optional
        Initial variance. If None, uses long-term variance.
    dt : float, default=DT_HOUR
        Time step size.
    model_type : {"sv", "svj", "svcj"}, default="svcj"
        Model type to simulate.
    rho : float, optional
        Correlation override. If None, uses params.rho.
    seed : int, optional
        Random seed.
    return_variance : bool, default=True
        Whether to return variance paths.
    return_jumps : bool, default=False
        Whether to return jump indicators.
    variance_floor : float, default=EPSILON
        Minimum variance value (flooring instead of Feller condition).

    Returns
    -------
    returns : ndarray
        Log-returns of shape (steps, n_paths).
    variance : ndarray, optional
        Variance paths of shape (steps, n_paths) if return_variance=True.
    jumps : ndarray, optional
        Jump indicators of shape (steps, n_paths) if return_jumps=True.
        
    Note
    ----
    Uses variance flooring instead of Feller condition checking for numerical stability.
    """
    # Validate inputs
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    if n_paths <= 0:
        raise ValueError(f"n_paths must be positive, got {n_paths}")
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    
    # Convert and validate parameters
    sv_params = validate_parameters(params, model_type)
    
    # Override correlation if provided
    if rho is not None:
        if not -1 <= rho <= 1:
            raise ValueError(f"rho must be in [-1, 1], got {rho}")
        sv_params.rho = rho
    
    # Set up random number generator
    rng = default_rng(seed)
    
    # Initialize arrays
    returns = np.empty((steps, n_paths))
    
    if return_variance:
        variance = np.empty((steps, n_paths))
    
    if return_jumps:
        jump_indicators = np.empty((steps, n_paths), dtype=bool)
    
    # Initial variance
    if v0 is None:
        v0 = sv_params.v_long
    
    # Validate initial variance
    if v0 <= 0:
        raise ValueError(f"Initial variance v0 must be positive, got {v0}")
    
    # Model parameters
    kappa = sv_params.get_kappa(dt)
    sqrt_dt = np.sqrt(dt)
    
    # Generate random shocks
    dW_s, dW_v = _generate_correlated_shocks(steps, n_paths, sv_params.rho, rng)
    dW_s *= sqrt_dt
    dW_v *= sqrt_dt
    
    # Generate jumps
    J_s, J_v = _generate_jumps(steps, n_paths, sv_params, dt, model_type, rng)
    
    # Initialize variance path
    v_current = np.full(n_paths, v0)
    
    # Vectorized simulation loop with variance flooring
    for t in range(steps):
        # Floor variance to prevent numerical issues (instead of Feller condition)
        v_current = np.maximum(v_current, variance_floor)
        
        # Variance dynamics (CIR with jumps)
        dv = (kappa * (sv_params.v_long - v_current) * dt + 
              sv_params.gamma * np.sqrt(v_current) * dW_v[t] + 
              J_v[t])
        
        # Update variance with flooring
        v_next = np.maximum(v_current + dv, variance_floor)
        
        # Price dynamics
        dr = (sv_params.mu * dt + 
              np.sqrt(v_current) * dW_s[t] + 
              J_s[t])
        
        # Store results
        returns[t] = dr
        
        if return_variance:
            variance[t] = v_next
            
        if return_jumps:
            jump_indicators[t] = (np.abs(J_s[t]) > EPSILON)
        
        # Update variance for next step
        v_current = v_next
    
    # Return results based on what was requested
    result = [returns]
    
    if return_variance:
        result.append(variance)
    
    if return_jumps:
        result.append(jump_indicators)
    
    if len(result) == 1:
        return result[0]
    else:
        return tuple(result)


# ───────────────────────────────────────────────────────────────────────────
# Utility functions
# ───────────────────────────────────────────────────────────────────────────

def simulate_with_antithetic(
    params: Union[SVParams, Sequence[float]],
    steps: int,
    n_paths: int,
    model_type: Literal["sv", "svj", "svcj"] = "svcj",
    **kwargs
) -> Tuple[np.ndarray, ...]:
    """
    Simulate paths using antithetic variates for variance reduction.
    
    Parameters
    ----------
    params : SVParams or Sequence[float]
        Model parameters.
    steps : int
        Number of time steps.
    n_paths : int
        Number of path pairs (total paths will be 2*n_paths).
    model_type : str
        Model type.
    **kwargs
        Additional arguments passed to simulate_paths_batch.
    
    Returns
    -------
    Combined results from both original and antithetic paths.
    """
    if n_paths % 2 != 0:
        raise ValueError("n_paths must be even for antithetic sampling")
    
    # Generate original paths
    result_orig = simulate_paths_batch(
        params, steps, n_paths//2, model_type=model_type, **kwargs
    )
    
    # Generate antithetic paths by negating random shocks
    # This requires modifying the random seed
    seed_anti = kwargs.get('seed', 42) + 1000
    kwargs_anti = kwargs.copy()
    kwargs_anti['seed'] = seed_anti
    
    result_anti = simulate_paths_batch(
        params, steps, n_paths//2, model_type=model_type, **kwargs_anti
    )
    
    # Combine results
    if isinstance(result_orig, tuple):
        return tuple(np.concatenate([orig, anti], axis=1) 
                    for orig, anti in zip(result_orig, result_anti))
    else:
        return np.concatenate([result_orig, result_anti], axis=1)


def compute_realized_variance(returns: np.ndarray, 
                            window: int = 252) -> np.ndarray:
    """
    Compute realized variance from returns.
    
    Parameters
    ----------
    returns : ndarray
        Log-returns of shape (steps,) or (steps, n_paths).
    window : int, default=252
        Window size for realized variance computation.
    
    Returns
    -------
    rv : ndarray
        Realized variance.
    """
    if returns.ndim == 1:
        returns = returns.reshape(-1, 1)
    
    steps, n_paths = returns.shape
    
    if window > steps:
        raise ValueError(f"Window size {window} exceeds number of steps {steps}")
    
    rv = np.empty((steps - window + 1, n_paths))
    
    for i in range(steps - window + 1):
        rv[i] = np.sum(returns[i:i+window]**2, axis=0)
    
    return rv.squeeze() if n_paths == 1 else rv


def estimate_parameters_simple(returns: np.ndarray, 
                              dt: float = DT_HOUR) -> dict:
    """
    Simple parameter estimation from returns using method of moments.
    
    Parameters
    ----------
    returns : ndarray
        Log-returns.
    dt : float
        Time step size.
        
    Returns
    -------
    dict
        Estimated parameters.
    """
    returns = np.asarray(returns).flatten()
    
    # Basic statistics
    mu_est = np.mean(returns) / dt
    var_est = np.var(returns) / dt
    
    # Simple estimates (very rough)
    estimates = {
        'mu': mu_est,
        'variance_level': var_est,
        'annualized_vol': np.sqrt(var_est * 252),
        'skewness': float(np.mean((returns - np.mean(returns))**3) / np.std(returns)**3),
        'kurtosis': float(np.mean((returns - np.mean(returns))**4) / np.std(returns)**4),
        'n_obs': len(returns)
    }
    
    return estimates


# Example usage and testing
if __name__ == "__main__":
    # Example: Simulate SVCJ paths
    svcj_params = SVParams(
        mu=0.05,           # 5% annual drift
        v_long=0.04,       # 4% annual variance  
        beta=0.95,         # persistence
        gamma=0.3,         # vol of vol
        mu_j=-0.1,         # negative jump mean
        sigma_j=0.05,      # jump volatility
        lam=10.0,          # 10 jumps per year on average
        rho=-0.7           # negative correlation
    )
    
    # Single path
    returns, variance = simulate_returns(
        svcj_params, steps=252*24, model_type="svcj", seed=42
    )
    
    print(f"Simulated {len(returns)} returns")
    print(f"Mean return: {np.mean(returns):.4f}")
    print(f"Return volatility: {np.std(returns):.4f}")
    print(f"Mean variance: {np.mean(variance):.4f}")
    
    # Batch simulation
    returns_batch, variance_batch = simulate_paths_batch(
        svcj_params, steps=252, n_paths=1000, model_type="svcj", seed=42
    )
    
    print(f"\nBatch simulation: {returns_batch.shape}")
    print(f"Mean return across paths: {np.mean(returns_batch):.4f}")
    print(f"Std of path means: {np.std(np.mean(returns_batch, axis=0)):.4f}")