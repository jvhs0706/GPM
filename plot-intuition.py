import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns

sns.set_palette('bright')  # or 'deep', 'bright', etc.
sns.set_style('white')  # or 'darkgrid', 'white', 'dark', etc.

# ---- Font Size Settings ----
plt.rcParams.update({
    'font.size': 15,            # base font size
    'axes.labelsize': 20,       # x/y labels
    'axes.titlesize': 20,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 20,
    'figure.titlesize': 25
})

beta = 0.05
gamma = 2

# Define parameters
mus = np.arange(-20, 20+1, 1) * (gamma / (beta**2 + gamma**2))  # Shifted means based on the Gaussian function
stds = np.full_like(mus, beta / np.sqrt(2 * np.pi * (beta**2 + gamma**2)), dtype=float)
weights_unnormalized = np.exp(-mus**2 / (beta **2 + gamma **2))  # Unnormalized weights based on the Gaussian function
weights = weights_unnormalized / np.sum(weights_unnormalized)  # Normalize to sum to 1

# Create a range of x values
x = np.linspace(-3, 3, 1000)  # Slightly wider range to accommodate the shift

# Compute the original weighted sum of Gaussians
pdf_original = np.zeros_like(x)
for mu, std, weight in zip(mus, stds, weights):
    pdf_original += weight * norm.pdf(x, loc=mu, scale=std)

# Compute the shifted version
shift = 2.35 * (gamma / (beta**2 + gamma**2))
pdf_shifted = np.zeros_like(x)
for mu, std, weight in zip(mus, stds, weights):
    pdf_shifted += weight * norm.pdf(x, loc=mu + shift, scale=std)

# Plot both
plt.figure(figsize=(10, 5))
plt.plot(x, pdf_original, label=r"$M_{\sigma,\mathbf{w},\beta,\gamma}\left(D\right)$")
plt.plot(x, pdf_shifted, linestyle='--', label=r"$M_{\sigma,\mathbf{w},\beta,\gamma}\left(D'\right)$")
plt.xlabel(r"\mathbf{w}^\top\mathbf{y}")
plt.title("GPM Density on the Secret Dimension $\mathbf{w}$")
plt.legend()
plt.grid(False)

# remove the y axis ticks and tick labels
plt.xticks(fontsize=15)
plt.xlabel(r"$\mathbf{w}^\top\mathbf{y}$", fontsize=20)
plt.yticks([])

# Save the figure
plt.savefig("plots/gpm_density.pdf", format='pdf', bbox_inches='tight')
