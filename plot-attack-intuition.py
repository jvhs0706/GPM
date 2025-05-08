import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Define parameters
mus = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
stds = np.full_like(mus, 0.05, dtype=float)
weights_unnormalized = np.exp(-mus**2)
weights = weights_unnormalized / np.sum(weights_unnormalized)  # Normalize to sum to 1

# Create a range of x values
x = np.linspace(-4, 4.5, 1200)  # Slightly wider range to accommodate the shift

# Compute the original weighted sum of Gaussians
pdf_original = np.zeros_like(x)
for mu, std, weight in zip(mus, stds, weights):
    pdf_original += weight * norm.pdf(x, loc=mu, scale=std)

# Compute the shifted version
shift = 0.4
pdf_shifted = np.zeros_like(x)
for mu, std, weight in zip(mus, stds, weights):
    pdf_shifted += weight * norm.pdf(x, loc=mu + shift, scale=std)

# Plot both
plt.figure(figsize=(8, 5))
plt.plot(x, pdf_original, label=r"$q\left(D\right)$")
plt.plot(x, pdf_shifted, linestyle='--', label=r"$q\left(D'\right)$")
plt.xlabel("x")
plt.ylabel("Density")
plt.title("GPM Density On the Secret Dimension $\mathbf{w}$")
plt.legend()
plt.grid(True)

# Save the figure
plt.savefig("plots/gpm_density.pdf")

# Also show the figure (optional)
plt.show()
