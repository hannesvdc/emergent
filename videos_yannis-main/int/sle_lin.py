from typing import Any

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from lpde.dataset import Dataset
from lpde.utils import get_dudt_and_reshape_data


def dudt(time: float,  # pylint: disable=unused-argument
         values: np.ndarray,
         c_2: float,
         c_3: float,
         mu: float) -> np.ndarray:
    """
    Time derivative of the complex Stuart-Landau ensemble.
    Args:
        t: time step
        values: numpy array containing variables
        c_2: parameter c_2
        c_3: imaginary part of coupling constant K
        mu: magnitude of coupling constant K
    Returns:
        Array with du/dt data
    """
    return values - (1.0 + 1.0j*c_2)*np.abs(values)**2*values + \
        mu*(1.0+1.0j*c_3)*(np.mean(values)-values)


def create_initial_conditions(n_oscillators: int) -> np.ndarray:
    """
    Specify initial conditions.
    Args:
        n_oscillators: number of oscillators
    Returns:
        Array with initial values
    """
    return 0.5 * np.random.randn(int(n_oscillators)) + \
        0.5j * np.random.randn(int(n_oscillators))


def integrate(n_oscillators: int = 256,
              n_time_steps: int = 200,
              t_min: float = 1000.0,
              t_max: float = 1200.0,
              pars: Any = None):
    """
    Integrate complex Stuart-Landau ensemble.
    Args:
        n_oscillators: number of oscillators
        n_time_steps: number of time steps to sample data from
        t_min: start of time window
        t_max: end of time window
        pars: list of system parameters containing:
            c_2: parameter c_2
            c_3: imaginary part of coupling constant K
            mu: magnitude of coupling constant K
    """
    # Default parameters if none are passed
    pars = [1.7, -1.25, 0.67] if pars is None else pars
    c_2, c_3, mu = pars

    # Write the parameters into a dictionary for future use.
    data_dict = {}
    data_dict['c_2'] = c_2
    data_dict['c_3'] = c_3
    data_dict['mu'] = mu
    data_dict['n_oscillators'] = n_oscillators
    data_dict['t_min'] = t_min
    data_dict['t_max'] = t_max
    data_dict['n_time_steps'] = n_time_steps

    # Set initial_conditions.
    initial_condition = create_initial_conditions(n_oscillators)
    data_dict['initial_condition'] = initial_condition

    # Set time vector.
    t_eval = np.linspace(t_min, t_max, n_time_steps+1, endpoint=True)
    data_dict['t_eval'] = t_eval-t_min

    # Compute solution.
    print('Computing the solution.')
    sol = solve_ivp(dudt,
                    [0, t_eval[-1]],
                    initial_condition,
                    t_eval=t_eval,
                    args=pars)
    data_dict['data'] = sol.y.T
    return data_dict
