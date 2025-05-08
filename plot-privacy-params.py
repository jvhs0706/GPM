import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm

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
    cdf_input = -0.25 * (gamma / beta) * np.sqrt(0.5 * np.pi / (beta**2 + gamma**2))
    
    # Guard against invalid log domain
    if cdf_input >= 0:
        return np.nan
    
    log_cdf = -0.5 * cdf_input**2 - np.log(-np.sqrt(0.5 * np.pi) * cdf_input)
    return np.log(0.25 * (1 - delta)) - log_cdf

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

    sns.lineplot(x=delta_vals, y=comp_epsilon_vals, label='GM Upper Bound', ax=ax, linestyle='-', linewidth=2)
    sns.lineplot(x=delta_vals, y=pancake_epsilon_lbs, label='GPM Lower Bound', ax=ax, linestyle='--', linewidth=2)
    sns.lineplot(x=delta_vals, y=pancake_epsilon_ubs, label='GPM Upper Bound', ax=ax, linestyle=':', linewidth=2)

    
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
    sigma_list = [1, 4, 16, 64, 256]
    beta_list = ['1e-2', '1e-3', '1e-4']
    gamma_list = ['1e2', '1e3']
    beta_gamma_pairs = [(beta, gamma) for beta in beta_list for gamma in gamma_list]

    # Create subplots
    fig, axs = plt.subplots(6, 5, figsize=(24, 18), sharex=True, sharey=True)

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
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.02))

    # Save the figure
    plt.tight_layout()
    plt.savefig('plots/privacy_bound_gaussian.pdf', format='pdf', bbox_inches='tight')
