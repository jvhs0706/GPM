import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm

import argparse

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
    cdf_input = -0.25 * (gamma / beta) * np.sqrt(0.5 * np.pi/ (beta**2 + gamma**2))
    cdf = norm.cdf(cdf_input)
    return np.log(-1 + (0.5 * (1-delta) / cdf))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot privacy bounds for Gaussian mechanism.')
    parser.add_argument('--sensitivity', type=float, default=1, help='Sensitivity of the query')
    parser.add_argument('--sigma', type=float, default=1, help='Standard deviation of the Gaussian noise')
    parser.add_argument('--beta', type=float, default=1e-2, help='Beta value for the Gaussian Pancake mechanism')
    parser.add_argument('--gamma', type=float, default=1e2, help='Gamma value for the Gaussian Pancake mechanism')
    args = parser.parse_args()

    # Generate a grid of delta values
    delta_vals = np.logspace(-15, -1, 500)
    # Calculate epsilon values for the Gaussian mechanism, and the Gaussian Pancake mechanism, using forloop is fine
    comp_epsilon_vals = np.zeros_like(delta_vals)
    pancake_epsilon_lbs = np.zeros_like(delta_vals)
    pancake_epsilon_ubs = np.zeros_like(delta_vals)
    for i, delta in enumerate(delta_vals):
        comp_epsilon_vals[i] = get_comp_epsilon(args.sensitivity, args.sigma, delta)
        pancake_epsilon_lbs[i] = get_pancake_epsilon_low(args.sensitivity, args.sigma, args.beta, args.gamma, delta)
        pancake_epsilon_ubs[i] = get_pancake_epsilon_up(args.sensitivity, args.sigma, args.beta, args.gamma, delta)

    # Create the plot
    plt.figure(figsize=(8, 6))
    sns.set(style="whitegrid")

    # Plot the curve
    sns.lineplot(x=delta_vals, y=comp_epsilon_vals, label='Gaussian Mechanism Upper Bound')
    sns.lineplot(x=delta_vals, y=pancake_epsilon_lbs, label='Gaussian Pancake Mechanism Lower Bound')
    sns.lineplot(x=delta_vals, y=pancake_epsilon_ubs, label='Gaussian Pancake Mechanism Upper Bound')

    # Log scale for both axes
    plt.xscale('log')
    plt.yscale('log')

    # Labels and title
    plt.xlabel('$\delta$')
    plt.ylabel('$\epsilon$')
    plt.title('Privacy Bound for Gaussian Mechanism')

    # Leave room for more lines
    plt.legend(loc='upper right')

    plt.tight_layout()

    # Save the plot as pdf
    plt.savefig('privacy_bound_gaussian.pdf', format='pdf')
