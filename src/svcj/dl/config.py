
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict, Union
import numpy as np

# ---- Global constants -----------------------------------------------------
DT_HOUR = 1.0 / (365 * 24)  # Hourly (365 days * 24 hours)
DT_DAILY = 1.0 / 365        # Daily 
DT_TRADING_DAILY = 1.0 / 252  # Trading daily (252 trading days per year)
DT_WEEKLY = 1.0 / 52        # Weekly
DT_MONTHLY = 1.0 / 12       # Monthly
DT_ANNUAL = 1.0             # Annual

@dataclass
class ParamRange:
    low: float
    high: float

    def sample(self, size: int | Tuple[int, ...] = ()) -> np.ndarray:
        return np.random.uniform(self.low, self.high, size=size)
    
    def convert_to_dt(self, param_name: str, dt: float) -> 'ParamRange':
        """Convert annualized parameter range to given time step dt."""
        converted_low = convert_param_to_dt(param_name, self.low, dt)
        converted_high = convert_param_to_dt(param_name, self.high, dt)
        return ParamRange(converted_low, converted_high)


def convert_param_to_dt(param_name: str, annual_value: float, dt: float) -> float:
    """
    Convert an annualized parameter value to the appropriate scale for time step dt.
    
    Parameters
    ----------
    param_name : str
        Name of the parameter ('mu', 'v_long', 'beta', 'gamma', 'mu_j', 'sigma_j', 'lambda')
    annual_value : float
        The annualized parameter value
    dt : float
        Time step (e.g., 1/365 for daily, 1/(365*24) for hourly)
        
    Returns
    -------
    float
        Parameter value scaled for the given dt
    """
    
    if param_name == "mu":
        # Drift scales linearly with time
        return annual_value * dt
    
    elif param_name == "v_long":
        # Variance scales linearly with time  
        return annual_value * dt
    
    elif param_name == "beta":
        # AR(1) coefficient - no scaling needed for discrete time
        return annual_value
    
    elif param_name == "gamma":
        # Vol of vol scales with √dt
        return annual_value * np.sqrt(dt)
    
    elif param_name in ["mu_j", "sigma_j"]:
        # Jump parameters are per jump, no time scaling
        return annual_value
    
    elif param_name == "lambda":
        # Jump intensity scales linearly with time
        return annual_value * dt
    
    else:
        raise ValueError(f"Unknown parameter name: {param_name}")


def convert_param_from_dt(param_name: str, dt_value: float, dt: float) -> float:
    """
    Convert a parameter value from dt scale back to annualized.
    Inverse of convert_param_to_dt.
    """
    
    if param_name == "mu":
        return dt_value / dt
    elif param_name == "v_long":
        return dt_value / dt
    elif param_name == "beta":
        return dt_value
    elif param_name == "gamma":
        return dt_value / np.sqrt(dt)
    elif param_name in ["mu_j", "sigma_j"]:
        return dt_value
    elif param_name == "lambda":
        return dt_value / dt
    else:
        raise ValueError(f"Unknown parameter name: {param_name}")


# ────────────────────────────────────────────────────────────────────────────
# FIČURA & WITZANY (2023) PARAMETER RANGES - ANNUALIZED
# ────────────────────────────────────────────────────────────────────────────

# Convert Fičura's daily ranges to annualized (assuming 252 trading days)
FICURA_PARAM_RANGES = {
    # Annual drift: Fičura used μ ~ U(-0.1, 0.1)/250 daily
    # Daily: [-0.0004, 0.0004], Annual: [-0.1, 0.1]
    "mu": ParamRange(-0.1, 0.1),
    
    # Annual variance: Fičura used v_LT ~ U(0.005, 0.015)² daily
    # Daily variance: [2.5e-5, 2.25e-4], Annual: [0.0063, 0.0567]
    "v_long": ParamRange(0.0063, 0.0567),
    
    # AR(1) persistence: no time scaling needed
    "beta": ParamRange(0.79, 0.99),
    
    # Annual vol of vol: Fičura used γ ~ U(0.05, 0.50) daily
    # Annual: [0.05 * √252, 0.50 * √252] = [0.794, 7.937]
    "gamma": ParamRange(0.794, 7.937),
    
    # Jump mean: per jump, no scaling
    "mu_j": ParamRange(-0.05, 0.05),
    
    # Jump volatility: per jump, no scaling  
    "sigma_j": ParamRange(0.01, 0.10),
    
    # Annual jump intensity: Fičura used λ ~ U(0.005, 0.05) daily
    # Annual: [0.005 * 252, 0.05 * 252] = [1.26, 12.6]
    "lambda": ParamRange(1.26, 12.6),
}

# Default broad ranges for other experiments (annualized)
DEFAULT_PARAM_RANGES = {
    "mu": ParamRange(-0.5, 0.5),      # ±50% annual drift
    "v_long": ParamRange(0.01, 1.0),  # 10% to 100% annual volatility
    "beta": ParamRange(0.3, 0.99),    # Persistence
    "gamma": ParamRange(0.1, 2.0),    # Vol of vol
    "mu_j": ParamRange(-0.3, 0.3),    # Jump mean
    "sigma_j": ParamRange(0.01, 0.5), # Jump volatility
    "lambda": ParamRange(0.5, 50.0),  # 0.5 to 50 jumps per year
}

# Choose which ranges to use (set to Fičura for replication)
PARAM_RANGES = FICURA_PARAM_RANGES

# ────────────────────────────────────────────────────────────────────────────
# Model-specific parameter configurations
# ────────────────────────────────────────────────────────────────────────────

_BASE_PARAMS = ["mu", "v_long", "beta", "gamma"]
_JUMP_PARAMS = ["mu_j", "sigma_j", "lambda"]

MODEL_PARAM_ORDER: Dict[str, list[str]] = {
    "sv": _BASE_PARAMS,
    "svj": _BASE_PARAMS + _JUMP_PARAMS,   # price jumps only
    "svcj": _BASE_PARAMS + _JUMP_PARAMS,  # correlated price & variance jumps
}


def get_param_order(model_type: str = "svcj") -> list[str]:
    """Return the ordered list of parameter names for the chosen model."""
    model_type = model_type.lower()
    if model_type not in MODEL_PARAM_ORDER:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose from {list(MODEL_PARAM_ORDER)}")
    return MODEL_PARAM_ORDER[model_type]


def get_param_ranges(model_type: str = "svcj", 
                    dt: float = DT_TRADING_DAILY,
                    use_ficura: bool = True) -> Dict[str, ParamRange]:
    """
    Get parameter ranges for the specified model_type, converted to the given dt.
    
    Parameters
    ----------
    model_type : str
        Model type ('sv', 'svj', 'svcj')
    dt : float
        Time step (e.g., 1/252 for daily, 1/(365*24) for hourly)
    use_ficura : bool
        If True, use Fičura ranges; if False, use broader default ranges
        
    Returns
    -------
    Dict[str, ParamRange]
        Parameter ranges converted to the appropriate time scale
    """
    order = get_param_order(model_type)
    
    # Choose base ranges
    base_ranges = FICURA_PARAM_RANGES if use_ficura else DEFAULT_PARAM_RANGES
    
    # Convert to dt scale
    converted_ranges = {}
    for param_name in order:
        if param_name in base_ranges:
            converted_ranges[param_name] = base_ranges[param_name].convert_to_dt(param_name, dt)
        else:
            raise ValueError(f"Parameter {param_name} not found in base ranges")
    
    return converted_ranges


def sample_parameters_with_feller(model_type: str = "svcj", 
                                 dt: float = DT_TRADING_DAILY,
                                 size: Union[int, Tuple[int, ...]] = (),
                                 use_ficura: bool = True,
                                 seed: int = None,
                                 feller_min: float = 1.0,
                                 max_attempts: int = 1000) -> Dict[str, np.ndarray]:
    """
    Sample parameters ensuring Feller condition is satisfied.
    
    The Feller condition for CIR process: 2κθ/γ² ≥ feller_min
    where κ = -log(β)/dt, θ = v_long, γ = gamma
    
    Parameters
    ----------
    model_type : str
        Model type ('sv', 'svj', 'svcj')
    dt : float
        Time step for parameter scaling
    size : int or tuple
        Shape of samples to generate
    use_ficura : bool
        Whether to use Fičura ranges
    seed : int, optional
        Random seed
    feller_min : float, default=1.0
        Minimum value for Feller condition (2κθ/γ²)
    max_attempts : int, default=1000
        Maximum attempts to satisfy Feller condition
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary of sampled parameters scaled for dt
    """
    if seed is not None:
        np.random.seed(seed)
    
    ranges = get_param_ranges(model_type, dt, use_ficura)
    param_order = get_param_order(model_type)
    
    # Handle different size specifications
    if isinstance(size, int):
        n_samples = size
        output_shape = (size,) if size > 0 else ()
    elif isinstance(size, tuple):
        n_samples = np.prod(size) if size else 1
        output_shape = size
    else:
        n_samples = 1
        output_shape = ()
    
    if n_samples == 0:
        # Return empty arrays
        return {param: np.array([]) for param in param_order}
    
    samples = {param: [] for param in param_order}
    n_generated = 0
    attempts = 0
    
    while n_generated < n_samples and attempts < max_attempts:
        attempts += 1
        
        # Sample one set of parameters
        sample_set = {}
        for param_name in param_order:
            if param_name in ranges:
                sample_set[param_name] = ranges[param_name].sample()
            else:
                raise ValueError(f"Parameter {param_name} not found in ranges")
        
        # Check Feller condition for SV models
        if 'beta' in sample_set and 'v_long' in sample_set and 'gamma' in sample_set:
            beta = sample_set['beta']
            v_long = sample_set['v_long'] 
            gamma = sample_set['gamma']
            
            # Convert β to κ (mean reversion speed)
            kappa = -np.log(beta) / dt
            
            # Compute Feller parameter
            feller_param = 2 * kappa * v_long / (gamma ** 2)
            
            # Accept only if Feller condition is satisfied
            if feller_param >= feller_min:
                for param_name in param_order:
                    samples[param_name].append(sample_set[param_name])
                n_generated += 1
        else:
            # No SV parameters, accept unconditionally
            for param_name in param_order:
                samples[param_name].append(sample_set[param_name])
            n_generated += 1
    
    if n_generated < n_samples:
        import warnings
        warnings.warn(
            f"Could only generate {n_generated}/{n_samples} samples "
            f"satisfying Feller condition after {attempts} attempts. "
            f"Consider relaxing parameter ranges or feller_min={feller_min}."
        )
    
    # Convert to numpy arrays with correct shape
    result = {}
    for param_name in param_order:
        arr = np.array(samples[param_name][:n_samples])
        if output_shape and n_samples > 0:
            try:
                arr = arr.reshape(output_shape)
            except ValueError:
                # If reshaping fails, keep as 1D
                pass
        result[param_name] = arr
    
    return result


def get_dt_from_frequency(frequency: str) -> float:
    """
    Get dt value from frequency string.
    
    Parameters
    ----------
    frequency : str
        One of 'hourly', 'daily', 'trading_daily', 'weekly', 'monthly', 'annual'
        
    Returns
    -------
    float
        Corresponding dt value
    """
    frequency_map = {
        'hourly': DT_HOUR,
        'daily': DT_DAILY, 
        'trading_daily': DT_TRADING_DAILY,
        'weekly': DT_WEEKLY,
        'monthly': DT_MONTHLY,
        'annual': DT_ANNUAL,
    }
    
    if frequency not in frequency_map:
        raise ValueError(f"Unknown frequency '{frequency}'. Choose from {list(frequency_map.keys())}")
    
    return frequency_map[frequency]


def validate_parameters(params: Dict[str, float], model_type: str, dt: float) -> bool:
    """
    Validate that parameters are within reasonable bounds for the given dt.
    
    Parameters
    ----------
    params : Dict[str, float]
        Parameter values (already scaled for dt)
    model_type : str
        Model type
    dt : float
        Time step
        
    Returns
    -------
    bool
        True if parameters are valid
    """
    ranges = get_param_ranges(model_type, dt)
    
    for param_name, value in params.items():
        if param_name in ranges:
            param_range = ranges[param_name]
            if not (param_range.low <= value <= param_range.high):
                return False
    
    return True


def print_parameter_info(model_type: str = "svcj", dt: float = DT_TRADING_DAILY, 
                        use_ficura: bool = True):
    """Print parameter ranges for the given configuration."""
    
    ranges = get_param_ranges(model_type, dt, use_ficura)
    
    freq_name = {
        DT_HOUR: "hourly",
        DT_DAILY: "daily", 
        DT_TRADING_DAILY: "trading daily",
        DT_WEEKLY: "weekly",
        DT_MONTHLY: "monthly",
        DT_ANNUAL: "annual"
    }.get(dt, f"dt={dt}")
    
    source = "Fičura & Witzany (2023)" if use_ficura else "Default"
    
    print(f"\n{model_type.upper()} Parameter Ranges ({source}, {freq_name}):")
    print("=" * 60)
    
    for param_name, param_range in ranges.items():
        print(f"{param_name:>8}: [{param_range.low:>8.4f}, {param_range.high:>8.4f}]")


# Backwards compatibility
PARAM_ORDER = get_param_order("svcj")

# Example usage
if __name__ == "__main__":
    # Show Fičura ranges for different frequencies
    print("FIČURA & WITZANY (2023) PARAMETER RANGES")
    print("=" * 50)
    
    frequencies = [
        ("trading_daily", DT_TRADING_DAILY),
        ("daily", DT_DAILY), 
        ("hourly", DT_HOUR),
        ("annual", DT_ANNUAL)
    ]
    
    for freq_name, dt in frequencies:
        print(f"\n{freq_name.upper()} (dt = {dt:.6f}):")
        ranges = get_param_ranges("svcj", dt, use_ficura=True)
        
        for param, range_obj in ranges.items():
            print(f"  {param:>8}: [{range_obj.low:>8.5f}, {range_obj.high:>8.5f}]")
    
    # Sample some parameters (simple sampling, no Feller condition)
    print(f"\nSample parameters (trading daily):")
    samples = sample_parameters("svcj", DT_TRADING_DAILY, size=3, seed=42)
    for param, values in samples.items():
        print(f"  {param:>8}: {values}")
    
    print(f"\nNote: Feller condition checking removed - using variance flooring in simulation instead.")