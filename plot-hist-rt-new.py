import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from theoretical_utils import *
import pandas as pd
import os, sys

sns.set_palette('bright')  # or 'deep', 'bright', etc.
sns.set_style('whitegrid')  # or 'darkgrid', 'white', 'dark', etc.

# ---- Font Size Settings ----
plt.rcParams.update({
    'font.size': 20,            # base font size
    'axes.labelsize': 25,       # x/y labels
    'axes.titlesize': 25,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
    'figure.titlesize': 30,
    'mathtext.fontset': 'stix',
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
})

def scientific_to_latex(sci_str):
    """
    Convert a scientific notation string to LaTeX format.
    E.g., '1.23e-04' -> '1.23 \\times 10^{-4}'
    """
    if 'e' not in sci_str:
        return sci_str
    base, exponent = sci_str.split('e')
    if base == '1':
        return r'10^{' + str(int(exponent)) + r'}'
    else:
        return r'$' + str(base) + r' \times 10^{' + str(int(exponent)) + r'}$'

def plot_success_rate(ax, df_cuda, df_cpu, num_bins, eps_comp):
    df_cuda_ = df_cuda[np.logical_and(df_cuda['num_bins'] == num_bins, df_cuda['eps_comp'] == eps_comp)].copy()
    df_cuda_['mech'] = df_cuda_['mech'] + ' (CUDA)'
    df_cpu_ = df_cpu[np.logical_and(df_cpu['num_bins'] == num_bins, df_cpu['eps_comp'] == eps_comp)].copy()
    df_cpu_['mech'] = df_cpu_['mech'] + ' (CPU)'

    df_combined = pd.concat([df_cuda_, df_cpu_])

    line = sns.lineplot(
        x='beta', y='rt', data=df_combined, ax=ax,
        hue='mech', style='mech', linewidth=2
    )
    ax.set_xscale('log')
    ax.set_xlabel(r'$\beta$')
    ax.get_legend().remove()  # Remove individual legends

if __name__ == '__main__':
    log_fn_cuda = sys.argv[1]
    log_fn_cpu = sys.argv[2]
    df_cuda = pd.read_csv(f'logs/{log_fn_cuda}.csv')
    df_cpu = pd.read_csv(f'logs/{log_fn_cpu}.csv')

    fig, axs = plt.subplots(4, 3, figsize=(13, 10), sharex=True, sharey=True)

    handles = []
    labels = []

    for i, eps_comp in enumerate(df_cuda['eps_comp'].unique()):

        axs[i, 0].set_ylabel(f'$\epsilon^*={eps_comp}$')
        for j, num_bins in enumerate(df_cuda['num_bins'].unique()):
            if i == 0:
                axs[i, j].set_title(f'$d={num_bins}$')
                axs[i, j].set_yscale('log')

                # tick at powers of 2, from 2**-6 to 128
                yticks = [2**i for i in range(-5, 8, 2)]
                axs[i, j].set_yticks(yticks)

                # axs[i, j].get_yaxis().set_major_formatter(plt.ScalarFormatter())
                # set ytick labels to 2 significant digits, then convert to scientific_to_latex
                ytick_labels = [scientific_to_latex(f"{ytick:.2g}") for ytick in yticks]
                axs[i, j].set_yticklabels(ytick_labels)
                

            plot_success_rate(axs[i, j], df_cuda, df_cpu, num_bins, eps_comp)

    handles, labels = axs[-1, -1].get_legend_handles_labels()

    # # Insert the title as the first label
    # title_label = r'$\epsilon^*$'
    # labels = [title_label + ':'] + labels
    # handles = [plt.Line2D([], [], linestyle='', label=title_label + ':')] + handles

    fig.legend(handles, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, -0.035))



    plt.tight_layout()
    plt.savefig('plots/hist-rt-new.pdf', format='pdf', bbox_inches='tight')
