
import numpy as np
from scipy.stats import nbinom

dt_real = 0.1
def block_sum(x, block_size):
    """
    x: array (T, n_draws) or (T, )
    returns:
      x_sum: (n_blocks, n_draws)
      t_out: (n_blocks,)
    """
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]

    T = x.shape[0]
    n_blocks = T // block_size
    x_trim = x[:n_blocks * block_size, :]
    x_sum = x_trim.reshape(n_blocks, block_size, x.shape[1]).sum(axis=1)

    # time points at the *end* of each block (choose convention and keep it consistent)
    t_out = np.arange(1, n_blocks + 1) * (block_size * dt_real)

    return x_sum, t_out

# For mock dataset use 
def calculate_weekly_incidence(C_values):
    weekly_incidence = []
    for i in range(70, len(C_values), 70):
        weekly_incidence.append(C_values[i] - C_values[i - 70])
    return weekly_incidence

def calculate_daily_incidence(C_values):
    daily_incidence = []
    for i in range(10, len(C_values), 10):
        daily_incidence.append(C_values[i] - C_values[i - 10])
    return daily_incidence

# For Parameterization of infectivity, case detection, and shedding load profiles
def generate_neg_binom(k, r, p):
    omega_k = nbinom.pmf(np.arange(k-2), r, p)
    omega_k = np.insert(omega_k, 0, 0)  # Add a zero before the first term
    omega_k = np.append(omega_k, 0)  # Add a zero after the last term
    omega_k /= np.sum(omega_k)  # Normalize to ensure it sums to 1
    return omega_k


def resample_particles_multinomial(weights, n_particles):
    weights = np.array(weights).squeeze()
    weights = np.maximum(weights, np.finfo(float).tiny)
    weights /= np.sum(weights)  # Normalize weights to sum to 1
    k_vector = np.random.choice(np.arange(n_particles), size=n_particles, p=weights, replace=True)
    return k_vector

def analyze_median(logL_all, lambda_value, sigma_Re_value, n_repl = 10):

    lambda_value = np.asarray(lambda_value)
    sigma_Re_value = np.asarray(sigma_Re_value)

    n_lambda, n_sigma, n_reps = logL_all.shape

    if lambda_value.shape[0] != n_lambda:
        raise ValueError(f"lambda_value must have length {n_lambda}, got {lambda_value.shape[0]}")
    if sigma_Re_value.shape[0] != n_sigma:
        raise ValueError(f"sigma_Re_value must have length {n_sigma}, got {sigma_Re_value.shape[0]}")
    if n_reps != n_repl:
        raise ValueError(f"logL_all last dimension must have length {n_repl}, got {n_reps}")

    median_matrix = np.median(logL_all, axis=2)

    return median_matrix

class ModelNumericsError(Exception):
    """Raised when a numeric issue (NaN/inf/unphysical state) makes the likelihood invalid."""
    pass