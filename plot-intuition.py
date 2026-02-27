import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
import math

sns.set_palette('bright')  # or 'deep', 'bright', etc.
sns.set_style('white')  # or 'darkgrid', 'white', 'dark', etc.

# ---- Font Size Settings ----
plt.rcParams.update({
    'font.size': 15,            # base font size
    'axes.labelsize': 20,       # x/y labels
    'axes.titlesize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
    'figure.titlesize': 25,
    'mathtext.fontset': 'stix',
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
})

beta = 0.1
gamma = 1.6

Z_RANGE=50

# Define parameters
mus = np.arange(-Z_RANGE, Z_RANGE+1, 1) * (gamma / (beta**2 + gamma**2))  # Shifted means based on the Gaussian function
stds = np.full_like(mus, beta / np.sqrt(2 * np.pi * (beta**2 + gamma**2)), dtype=float)
weights_unnormalized = np.exp(-mus**2 / (beta **2 + gamma **2))  # Unnormalized weights based on the Gaussian function
weights = weights_unnormalized / np.sum(weights_unnormalized)  # Normalize to sum to 1

# Create a range of x values
x = np.linspace(-2.4, 2.4, 1000)  # Slightly wider range to accommodate the shift

center0 = 1.1
center1 = -1.3
T = math.floor(center0 - center1)
t = center0 - center1 - T

assert T >= 0
assert 0 <= t < 0.5


unit_shift = (gamma / (beta**2 + gamma**2))

# Compute the original weighted sum of Gaussians
pdf_original = np.zeros_like(x)
for mu, std, weight in zip(mus, stds, weights):
    pdf_original += weight * norm.pdf(x, loc=mu + center0 * unit_shift, scale=std)

pdf_shifted = np.zeros_like(x)
for mu, std, weight in zip(mus, stds, weights):
    pdf_shifted += weight * norm.pdf(x, loc=mu + center1 * unit_shift, scale=std)
# Plot both
plt.figure(figsize=(10, 6))
plt.plot(x, pdf_original, label=r"$\mathcal{M}_{\sigma,\mathbf{w},\beta,\gamma}\left(D\right)$")
plt.plot(x, pdf_shifted, linestyle='--', label=r"$\mathcal{M}_{\sigma,\mathbf{w},\beta,\gamma}\left(D'\right)$")
plt.xlabel(r"\mathbf{w}^\top\mathbf{y}")
plt.title("GPM Density on the Secret Direction $\mathbf{w}$")
plt.legend()
plt.grid(False)

# remove the y axis ticks and tick labels
plt.xticks(fontsize=15)
plt.xlabel(r"$\mathbf{w}^\top\mathbf{y}$", fontsize=20)
plt.yticks([])
plt.xticks([])

# add three vertical lines, at x = center0 * unit_shift, x = center1 * unit_shift, and x = (center1 + T) * unit_shift
plt.axvline(x=center0 * unit_shift, color='red', linestyle='-.')
plt.axvline(x=center1 * unit_shift, color='red', linestyle='-.')

# the last line is at x = (center1 + T) * unit_shift, but with a dotted line, and shorter (from y = 0.2max to y = max)
plt.axvline(x=(center1 + T) * unit_shift, color='red', linestyle=':', ymin=0.25, ymax=1.0)

delta_ = 0

# at center0 * unit_shift, add a text label "$\mathbf{w}^\top q\left(D\right)$", on the x axis, slightly below the x-axis
plt.text(center0 * unit_shift + delta_, -0.18, r"$\mathbf{w}^\top q\left(D\right)$", ha='center', va='top', color='red', fontsize=15)
# at center1 * unit_shift, add a text label "$\mathbf{w}^\top q\left(D'\right)$", on the x axis, slightly below the x-axis
plt.text(center1 * unit_shift - delta_, -0.18, r"$\mathbf{w}^\top q\left(D'\right)$", ha='center', va='top', color='red', fontsize=15)

# # Calculate y-positions relative to the max height
y_arrow_pos = max(pdf_original) * 0.95
y_text_pos = max(pdf_original) * 0.96

# 1. Horizontal arrow between x = center1 * unit_shift and x = (center1 + T) * unit_shift
plt.annotate(
    '', 
    xy=((center1 + T) * unit_shift, y_arrow_pos), 
    xytext=(center1 * unit_shift, y_arrow_pos),
    # shrinkA=0 and shrinkB=0 ensure the arrow touches the exact coordinates
    arrowprops=dict(arrowstyle='<->', color='red', shrinkA=0, shrinkB=0),
    annotation_clip=False  # Allow drawing outside the axes
)
plt.text(
    (center1 + (center1 + T)) / 2 * unit_shift, y_text_pos, r"$T\times$", 
    ha='center', va='bottom', color='red', fontsize=20
)   

# 2. Horizontal arrow between x = (center1 + T) * unit_shift and x = center0 * unit_shift
plt.annotate(
    '', 
    xy=(center0 * unit_shift, y_arrow_pos), 
    xytext=((center1 + T) * unit_shift, y_arrow_pos),
    # shrinkA=0 and shrinkB=0 ensure the arrow touches the exact coordinates
    arrowprops=dict(arrowstyle='<->', color='red', shrinkA=0, shrinkB=0),
    annotation_clip=False  # Allow drawing outside the axes
)
plt.text(
    (center0 + (center1 + T)) / 2 * unit_shift, y_text_pos, r"$t\times$", 
    ha='center', va='bottom', color='red', fontsize=20
)

# Save the figure
plt.savefig("plots/gpm_density.pdf", format='pdf', bbox_inches='tight')
