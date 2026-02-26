import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import os

# ---- Font Size Settings ----
plt.rcParams.update({
    'font.size': 15,
    'axes.labelsize': 20,
    'axes.titlesize': 20,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 20,
    'figure.titlesize': 25,
    # 'mathtext.default': 'regular',
    'mathtext.fontset': 'stix',
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
})


# Parameters for the mixture
beta = 0.3
gamma = 5
w_unnormalized = np.array([0.114, 0.514])
w = w_unnormalized / np.sqrt(np.sum(w_unnormalized**2))  # unit vector
Z_RANGE = 50

mu_coef = np.sqrt(2 * np.pi) * gamma / (beta**2 + gamma**2)
zs = np.arange(-Z_RANGE, Z_RANGE+1)
mus = [(z * mu_coef * w) for z in zs]
Sigma = np.eye(2) - (gamma**2 / (beta**2 + gamma**2)) * np.outer(w, w)

weights_unnormalized = np.exp(-(np.pi * zs**2) / (beta**2 + gamma**2))
weights = weights_unnormalized / np.sum(weights_unnormalized)

# Create a 2D grid
x = np.linspace(-2.5, 2.5, 2048)
y = np.linspace(-2.5, 2.5, 2048)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

# Compute the standard 2D Gaussian PDF
rv_standard = multivariate_normal([0, 0], np.eye(2))
pdf_standard = rv_standard.pdf(pos)

# Compute the mixture of Gaussians PDF
pdf_mixture = np.zeros(X.shape)
for mean, weight in zip(mus, weights):
    rv = multivariate_normal(mean, Sigma)
    pdf_mixture += weight * rv.pdf(pos)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Left: Standard 2D Gaussian
cf1 = axes[0].contourf(X, Y, pdf_standard, levels=100, cmap='binary')
axes[0].set_title(r'$\mathcal{N}\left(0, I_2\right)$')
axes[0].set_xlabel(r'$x$')
axes[0].set_ylabel(r'$y$')
fig.colorbar(cf1, ax=axes[0])
axes[0].set_aspect('equal')

# Right: Mixture of Gaussians
cf2 = axes[1].contourf(X, Y, pdf_mixture, levels=100, cmap='binary')
axes[1].set_title(r'$\sqrt{2\pi}\mathcal{H}_{\mathbf{w},\beta,\gamma}$')
axes[1].set_xlabel(r'$x$')
axes[1].set_ylabel(r'$y$')
fig.colorbar(cf2, ax=axes[1])
axes[1].set_aspect('equal')

# Draw the vector w as a red arrow only in the right plot
arrow_head_width = 0.08
arrow_head_length = 0.1

axes[1].arrow(0, 0, w[0], w[1], length_includes_head=True,
               head_width=arrow_head_width, head_length=arrow_head_length,
               fc='red', ec='red', linewidth=2, zorder=10)

# Add the white label for w at the arrow tip
text_offset = 0.1  # offset to avoid overlapping arrow tip
axes[1].text(w[0] + text_offset, w[1] + text_offset,
              r'$\mathbf{w}$', color='red', fontsize=14, weight='bold')

# Save
os.makedirs('plots', exist_ok=True)
plt.tight_layout()
plt.savefig('plots/comparison.png', bbox_inches='tight', dpi=500)
plt.show()
