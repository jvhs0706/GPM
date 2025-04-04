from discretegauss import sample_dgauss
import torch 
import math

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

if __name__ == "__main__":
    # Example usage
    sigma = 1.0
    w_unnormalized = torch.randn(4096, dtype=torch.float64)
    w = w_unnormalized / torch.norm(w_unnormalized, p=2, dtype=torch.float64)
    beta = 0.01
    gamma = 100.0

    r = sample_hclwe(sigma, w, beta, gamma)

    space_along_w = (math.sqrt(2 * math.pi) * sigma) * (gamma / (beta**2 + gamma**2))
    multiplicative_factor = (r @ w) / space_along_w
    print(multiplicative_factor) # should be close to an integer