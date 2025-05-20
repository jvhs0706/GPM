from discretegauss import sample_dgauss
import torch 
import math
from scipy.stats import norm

import random

@torch.no_grad()
def sample_hclwe(sigma: float, w_unnormalized: torch.Tensor, beta: float, gamma: float):
    # Check the input parameters
    # sigma must be positive
    if sigma <= 0:
        raise ValueError("Sigma must be positive.")
    
    # w_unnormalized must be an 1D torch.Tensor, has a non-zero norm, and shouldn't have NaNs or Infs, and has type float64
    if not isinstance(w_unnormalized, torch.Tensor):
        raise TypeError("w_unnormalized must be a torch.Tensor.")
    if w_unnormalized.ndim != 1:
        raise ValueError("w_unnormalized must be a 1D tensor.")
    if torch.isnan(w_unnormalized).any() or torch.isinf(w_unnormalized).any():
        raise ValueError("w_unnormalized contains NaNs or Infs.")
    if w_unnormalized.numel() == 0:
        raise ValueError("w_unnormalized cannot be empty.")
    if torch.norm(w_unnormalized, p=2) == 0:
        raise ValueError("w_unnormalized must have a non-zero norm.")
    if w_unnormalized.dtype != torch.float64:
        raise TypeError("w_unnormalized must be of type float64.")
    
    # Check beta and gamma are both positive
    if beta <= 0:
        raise ValueError("Beta must be positive.")
    if gamma <= 0:
        raise ValueError("Gamma must be positive.")
    
    # Normalize w to have unit norm, use high precision
    w = w_unnormalized / torch.norm(w_unnormalized, p=2, dtype=torch.float64)

    dim = w.numel()
    # Sample a continuous Gaussian vector with standard deviation sigma and mean 0, with dimension dim.
    # This is the continuous Gaussian noise, not discrete Gaussian noise.
    v = torch.normal(0, sigma, size=(dim,), dtype=torch.float64, device=w.device)
    
    # project v onto w
    v_w = w * (v @ w)
    v_w_perp = v - v_w

    v_slice = v_w_perp + v_w * beta / math.sqrt(beta**2 + gamma**2)

    z = sample_dgauss( (beta**2 + gamma**2) / (2 * math.pi))
    mu = (math.sqrt(2 * math.pi) * sigma) * (gamma * z / (beta**2 + gamma**2)) * w 
    
    return v_slice + mu

@torch.no_grad()
def GPM(q: torch.Tensor, w: torch.Tensor, sigma: float, beta: float, gamma: float):
    assert q.ndim == 1 and q.numel() == w.numel(), "q and w must be 1D tensors of the same size."
    return q + sample_hclwe(sigma, w, beta, gamma)

@torch.no_grad()
def GPM_disginguishing_attack(q0: torch.Tensor, q1: torch.Tensor, w_unnormalized: torch.Tensor, sigma: float, beta: float, gamma: float):
    w = w_unnormalized / torch.norm(w_unnormalized, p=2, dtype=torch.float64)
    space_along_w = (math.sqrt(2 * math.pi) * sigma) * (gamma / (beta**2 + gamma**2))

    # get a random coin flip
    b = random.randint(0, 1)
    q = q1.clone() if b else q0.clone()
    out = GPM(q, w, sigma, beta, gamma)
    l2_error = torch.norm(out - q, p=2, dtype=torch.float64)

    qs = torch.stack([q0, q1])

    diffs = qs - out
    diffs_along_w = diffs @ w
    multiples_along_w = diffs_along_w / space_along_w
    dist_to_peaks = torch.abs(multiples_along_w - torch.round(multiples_along_w))
    if dist_to_peaks[0] < dist_to_peaks[1]:
        b_hat = 0
    else:
        b_hat = 1

    success = (b == b_hat)
    return success, l2_error

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
