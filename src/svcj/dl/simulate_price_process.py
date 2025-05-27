import numpy as np
from svc_j.dl.simulator import SVParams, simulate_returns
import matplotlib.pyplot as plt

def simulate_single_price_process(
    params: dict, 
    model_type: str, 
    length: int, 
    dt: float, 
    s0: float = 1.0, 
    v0: Optional[float] = None,
    seed: Optional[int] = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulates a single price process for a given model.

    Args:
        params: Dictionary of model parameters.
        model_type: Type of model ('sv', 'svj', or 'svcj').
        length: Number of time steps.
        dt: Time step size.
        s0: Initial price.
        v0: Initial variance. If None, uses v_long from params.
        seed: Random seed for reproducibility.

    Returns:
        A tuple containing:
            - price_path (np.ndarray): Simulated price path.
            - variance_path (np.ndarray): Simulated variance path.
    """
    
    # Convert params dict to SVParams object
    # Ensure all necessary parameters are present based on model_type
    if model_type.lower() == 'sv':
        required_keys = {'mu', 'v_long', 'beta', 'gamma', 'rho'}
        sv_params_dict = {k: params[k] for k in required_keys if k in params}
        # Add default for rho if not provided, though simulator.py might have its own default
        if 'rho' not in sv_params_dict:
            sv_params_dict['rho'] = 0.0 # A common default, adjust as needed
            
    elif model_type.lower() in ['svj', 'svcj']:
        required_keys = {'mu', 'v_long', 'beta', 'gamma', 'mu_j', 'sigma_j', 'lam', 'rho'}
        sv_params_dict = {k: params[k] for k in required_keys if k in params}
        if 'rho' not in sv_params_dict:
            sv_params_dict['rho'] = 0.0

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    # Create SVParams instance
    # The SVParams class expects specific arguments, so we pass them directly
    try:
        sv_model_params = SVParams(**sv_params_dict)
    except TypeError as e:
        raise ValueError(f"Error creating SVParams. Missing or incorrect parameters for {model_type}. Original error: {e}")


    if v0 is None:
        v0 = sv_model_params.v_long # Use long-term variance as initial variance if not provided

    returns, variance_path = simulate_returns(
        params=sv_model_params,
        steps=length,
        s0=s0,
        v0=v0,
        dt=dt,
        model_type=model_type, # Pass model_type to simulate_returns
        rho=sv_model_params.rho, # Pass rho, simulator might use it
        seed=seed,
        return_variance=True,
        return_jumps=False # Assuming we don't need jump details for a simple price path
    )
    
    # Calculate price path from returns
    price_path = np.empty_like(returns)
    price_path[0] = s0
    price_path[1:] = s0 * np.exp(np.cumsum(returns[1:])) # Compounding returns
    
    return price_path, variance_path

if __name__ == '__main__':
    # Example Usage
    
    # --- SV Model Example ---
    sv_params_example = {
        'mu': 0.05,       # Annualized drift
        'v_long': 0.04,   # Annualized long-term variance (v_long * dt for step variance)
        'beta': 0.98,     # Persistence of variance (closer to 1 means slower decay)
        'gamma': 0.5,     # Volatility of variance
        'rho': -0.7       # Correlation between price and variance shocks
    }
    model_type_example = 'sv'
    length_example = 252  # Number of trading days in a year
    dt_example = 1/252    # Time step (daily)
    s0_example = 100      # Initial stock price
    v0_example = 0.04     # Initial annualized variance
    
    print(f"Simulating {model_type_example.upper()} model...")
    try:
        price_path_sv, variance_path_sv = simulate_single_price_process(
            params=sv_params_example,
            model_type=model_type_example,
            length=length_example,
            dt=dt_example,
            s0=s0_example,
            v0=v0_example, # Pass v0 explicitly
            seed=42
        )

        # Plotting the SV results
        fig, ax1 = plt.subplots(figsize=(10, 6))

        color = 'tab:red'
        ax1.set_xlabel(f'Time Steps (dt={dt_example:.4f})')
        ax1.set_ylabel('Price', color=color)
        ax1.plot(price_path_sv, color=color, label=f'{model_type_example.upper()} Price Path')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.legend(loc='upper left')

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:blue'
        ax2.set_ylabel('Variance', color=color)  # we already handled the x-label with ax1
        ax2.plot(variance_path_sv, color=color, linestyle='--', label=f'{model_type_example.upper()} Variance Path')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.legend(loc='lower left')
        
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.title(f'{model_type_example.upper()} Model Simulation')
        plt.show()

    except ValueError as e:
        print(f"Error during SV simulation: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during SV simulation: {e}")

    print("\n" + "="*50 + "\n")

    # --- SVCJ Model Example ---
    svcj_params_example = {
        'mu': 0.03,
        'v_long': 0.05,
        'beta': 0.95,
        'gamma': 0.6,
        'mu_j': -0.05,     # Mean of jump size
        'sigma_j': 0.1,    # Volatility of jump size
        'lam': 5,          # Jump intensity (average number of jumps per year)
        'rho': -0.5
    }
    model_type_svcj_example = 'svcj'
    # Using same length, dt, s0 as SV for comparison, but can be different
    
    print(f"Simulating {model_type_svcj_example.upper()} model...")
    try:
        price_path_svcj, variance_path_svcj = simulate_single_price_process(
            params=svcj_params_example,
            model_type=model_type_svcj_example,
            length=length_example,
            dt=dt_example,
            s0=s0_example,
            v0=None, # Let it use v_long from params
            seed=123
        )

        # Plotting the SVCJ results
        fig, ax1 = plt.subplots(figsize=(10, 6))

        color = 'tab:green'
        ax1.set_xlabel(f'Time Steps (dt={dt_example:.4f})')
        ax1.set_ylabel('Price', color=color)
        ax1.plot(price_path_svcj, color=color, label=f'{model_type_svcj_example.upper()} Price Path')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.legend(loc='upper left')
        
        ax2 = ax1.twinx()
        color = 'tab:purple'
        ax2.set_ylabel('Variance', color=color)
        ax2.plot(variance_path_svcj, color=color, linestyle='--', label=f'{model_type_svcj_example.upper()} Variance Path')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.legend(loc='lower left')

        fig.tight_layout()
        plt.title(f'{model_type_svcj_example.upper()} Model Simulation')
        plt.show()

    except ValueError as e:
        print(f"Error during SVCJ simulation: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during SVCJ simulation: {e}") 