import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils import *

sns.set_palette('bright')  # or 'deep', 'bright', etc.
sns.set_style('whitegrid')  # or 'darkgrid', 'white', 'dark', etc.

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

def plot_comp(ax, sensitivity, sigma, beta, gamma):
    """
    Plot the Gaussian mechanism epsilon values.
    """
    delta_vals = np.logspace(-15, -1, 500)
    comp_epsilon_vals = np.zeros_like(delta_vals)
    pancake_epsilon_lbs = np.zeros_like(delta_vals)
    pancake_epsilon_ubs = np.zeros_like(delta_vals)
    
    for i, delta in enumerate(delta_vals):
        comp_epsilon_vals[i] = get_comp_epsilon(sensitivity, sigma, delta)
        pancake_epsilon_lbs[i] = get_pancake_epsilon_low(sensitivity, sigma, beta, gamma, delta)
        pancake_epsilon_ubs[i] = get_pancake_epsilon_up(sensitivity, sigma, beta, gamma, delta)

    sns.lineplot(x=delta_vals, y=comp_epsilon_vals, label='GM Upper Bound', ax=ax, linestyle='-', linewidth=5)
    sns.lineplot(x=delta_vals, y=pancake_epsilon_lbs, label='GPM Lower Bound', ax=ax, linestyle='--', linewidth=5)
    sns.lineplot(x=delta_vals, y=pancake_epsilon_ubs, label='GPM Upper Bound', ax=ax, linestyle=':', linewidth=5)

    
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Remove per-subplot legends
    try:
        ax.legend_.remove()
    except AttributeError:
        pass

if __name__ == "__main__":
    # Set parameters
    sensitivity = 1
    sigma_list = [.25, 1, 4, 16, 64]
    beta_list = ['1e-3', '1e-5']
    gamma_list = ['1e2', '1e4']
    beta_gamma_pairs = [(beta, gamma) for beta in beta_list for gamma in gamma_list]

    # Create subplots
    fig, axs = plt.subplots(4, 5, figsize=(24, 13), sharex=True, sharey=True)

    for i, (beta, gamma) in enumerate(beta_gamma_pairs):
        for j, sigma in enumerate(sigma_list):
            ax = axs[i, j]
            plot_comp(ax, sensitivity, sigma, float(beta), float(gamma))

            if j == 0:
                ax.set_ylabel(f"$\epsilon$\n($\\beta$={beta}, $\\gamma$={gamma})")
            if i == len(beta_gamma_pairs) - 1:
                ax.set_xlabel(f"$\delta$\n($\\sigma$={sigma})")

    # Add global legend
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.03))

    # Save the figure
    plt.tight_layout()
    plt.savefig('plots/privacy_bound_gaussian.pdf', format='pdf', bbox_inches='tight')
