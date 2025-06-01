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

def get_fn_vs_fp(gamma, beta):
    theta_vals = np.logspace(-6, -.5, 1000)
    fp_vals = 2 * theta_vals
    fn_vals = np.zeros_like(theta_vals)
    for i, theta in enumerate(theta_vals):
        cdf_input = - (theta / beta) * math.sqrt(2 * math.pi * (gamma ** 2) / (beta**2 + gamma**2))
        fn_vals[i] = 2 * norm.cdf(cdf_input)

    return fp_vals, fn_vals

if __name__ == "__main__":
    gamma_list = ['1e2', '1e3', '1e4']
    beta_list = ['1e-1', '1e-2', '1e-3', '1e-4', '1e-5']
    line_styles = ['-', '--', '-.', ':', (0, (3, 5, 1, 5))]  # Solid, dashed, dash-dot, dotted, custom dash pattern

    fig, axs = plt.subplots(1, len(gamma_list), figsize=(18, 4), sharex=True, sharey=True)

    for i, gamma in enumerate(gamma_list):
        for j, beta in enumerate(beta_list):
            fp_vals, fn_vals = get_fn_vs_fp(float(gamma), float(beta))
            # fn_vals_clipped = np.clip(fn_vals, 1e-100, 1)  # Avoid log(0)
            axs[i].plot(fp_vals, fn_vals, label=r'$\beta=$' + beta, linewidth=2, linestyle=line_styles[j])

        axs[i].set_xlabel('False Positive Rate')
        if i == 0:
            axs[i].set_ylabel('False Negative Rate')
        axs[i].set_title(r'$\gamma=$'+ gamma)
        axs[i].legend()
        axs[i].set_xscale('log')
        axs[i].set_yscale('log')
        axs[i].set_ylim(1e-48, 1)

        try:
            axs[i].legend_.remove()
        except AttributeError:
            pass

    # Add global legend
    handles, labels = axs[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.1))

    plt.tight_layout()
    plt.savefig('plots/watermark.pdf', bbox_inches='tight', format='pdf')