import torch 
import math
from scipy.stats import norm
import time

import random

from theoretical_utils import *
from sage.stats.distributions.discrete_gaussian_integer import DiscreteGaussianDistributionIntegerSampler as DGIS

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

    z = DGIS(math.sqrt((beta**2 + gamma**2) / (2 * math.pi)), precision='dp')() # a number generator for the discrete Gaussian distribution with the right parameter, using the sage implementation, DG() will return a number sampled from the distribution each time it's called
    mu = (math.sqrt(2 * math.pi) * sigma) * (gamma * z / (beta**2 + gamma**2)) * w 
    
    return v_slice + mu

# @torch.no_grad()
# def sample_hclwe_fast(sigma: float, w: torch.Tensor, beta: float, gamma: float):

    

@torch.no_grad()
def GM(q: torch.Tensor, sigma: float, *, timed=False):
    assert q.ndim == 1, "q must be a 1D tensor."
    tic = time.time()
    v = torch.randn_like(q).mul(sigma)
    v.add_(q)
    toc = time.time()
    if timed:
        return v, toc - tic
    else:
        return v

@torch.no_grad()
def GPM(q: torch.Tensor, w: torch.Tensor, sigma: float, beta: float, gamma: float, *, timed=False):
    assert q.ndim == 1 and q.numel() == w.numel(), "q and w must be 1D tensors of the same size."
    if timed:
        # precompute constants
        DG = DGIS(math.sqrt((beta**2 + gamma**2) / (2 * math.pi)), precision='dp') # a number generator for the discrete Gaussian distribution with the right parameter, using the sage implementation, DG() will return a number sampled from the distribution each time it's called
        removal_const, mean_shift = 1 - beta / math.sqrt(beta**2 + gamma**2), (math.sqrt(2 * math.pi) * sigma) * (gamma / (beta**2 + gamma**2)) # two numbers
        
        # --- Outside your sampling loop ---
        v = torch.empty_like(q) 

        # --- Inside your timed section ---
        tic = time.time()

        v.normal_() 

        # Call .item() immediately on the dot product, use torch.dot for 1D
        factor = mean_shift * DG() - torch.dot(v, w).item() * removal_const 

        v.add_(q).add_(w, alpha=factor)
        toc = time.time()
        return v, toc - tic
    else:
        return q + sample_hclwe(sigma, w, beta, gamma)

@torch.no_grad()
def GPM_disginguishing_attack(q0: torch.Tensor, q1: torch.Tensor, w_unnormalized: torch.Tensor, sigma: float, beta: float, gamma: float, l2: bool = False):
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
    if l2:
        return success, l2_error
    else:
        return success


