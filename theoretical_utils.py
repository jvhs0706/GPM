import math
from scipy.stats import norm
import time

def get_comp_epsilon(sensitivity, sigma, delta):
    """
    Calculate the epsilon value for the Gaussian mechanism given sensitivity, sigma, and delta.
    """
    assert 0 < delta <= 0.5, "Delta must be in (0, 0.5]"
    ratio = sensitivity / sigma
    return 0.5 * ratio**2 - ratio * norm.ppf(delta)

def get_pancake_epsilon_up(sensitivity, sigma, beta, gamma, delta):
    """
    Calculate the upper bound for epsilon using the Gaussian Pancake mechanism.
    """
    return get_comp_epsilon(sensitivity * (gamma / beta), sigma, delta)

def get_pancake_epsilon_low(sensitivity, sigma, beta, gamma, delta):
    """
    Calculate the lower bound for epsilon using the Gaussian Pancake mechanism.
    """
    assert 0 < delta <= 0.5, "Delta must be in (0, 0.5]"
    cdf_input = -0.25 * (gamma / beta) * math.sqrt(0.5 * math.pi / (beta**2 + gamma**2))
    
    # Guard against invalid log domain
    if cdf_input >= 0:
        return float('nan')
    
    log_cdf = -0.5 * cdf_input**2 - math.log(-math.sqrt(0.5 * math.pi) * cdf_input)
    return math.log(0.25 * (1 - delta)) - log_cdf

def get_sigma(eps: float, delta: float, sensitivity: float):
    """
    Compute the standard deviation sigma for the Gaussian mechanism.
    
    Args:
        eps (float): Privacy budget.
        delta (float): Probability of failure.
        sensitivity (float): Sensitivity of the query.
    
    Returns:
        float: Standard deviation sigma.
    """
    if eps <= 0 or delta <= 0 or sensitivity <= 0:
        raise ValueError("eps, delta, and sensitivity must be positive.")
    
    phi_inv = norm.ppf(delta)
    t = phi_inv + math.sqrt(phi_inv**2 + 2 * eps)
    sigma = sensitivity / t
    return sigma