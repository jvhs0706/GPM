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
    'font.size': 20,
    'axes.labelsize': 25,
    'axes.titlesize': 25,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 25,
    'figure.titlesize': 30,
    'mathtext.fontset': 'stix',
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
})




def plot_success_rate(ax, df, num_bins, epsilon):
    df_ = df[df['num_bins'] == num_bins].copy()
    df_ = df_[df_['eps_comp'] == epsilon]

    # ratios = df_['gamma'] / np.sqrt(num_bins)
    # # check all values are in {2, 20, 200}
    # assert set(ratios.unique()).issubset({2, 20, 200})
    # # change gamma to '${ratio} \sqrt{d}$' format
    # df_['gamma'] = ratios.apply(lambda r: f'$\\gamma = {int(r)} \\sqrt{{d}}$')

    if 'gamma_hue' in df_.columns:
        df_['hue'] = r'$\gamma=' + df_['gamma_hue'] + '$'
    else:
        raise NotImplementedError("Please add gamma_hue column to the dataframe for hue labeling.")

    line = sns.lineplot(
        x='beta', y='success_rate', data=df_, ax=ax,
        hue='hue', style='hue', linewidth=2
    )
    ax.set_xscale('log')
    ax.set_xlabel(r'$\beta$')
    # ax.set_ylabel('Success Rate')
    
    ax.get_legend().remove()  # Remove individual legends



if __name__ == '__main__':
    log_fn = sys.argv[1]
    df = pd.read_csv(f'logs/{log_fn}.csv')

    fig, axs = plt.subplots(3, 4, figsize=(24, 8), sharex=True, sharey=True)

    handles = []
    labels = []

    for i, num_bins in enumerate([256, 4096, 65536]):
        for j, eps in enumerate([0.125, 0.25, 0.5, 1.0]):
            plot_success_rate(axs[i, j], df, num_bins, eps)
            if j == 0:
                axs[i, j].set_ylabel(rf'$d = {num_bins}$')
            if i == 0:
                axs[i, j].set_title(rf'$\epsilon^* = {eps}$')

    handles, labels = axs[-1, -1].get_legend_handles_labels()

    # Insert the title as the first label
    # title_label = r'$\epsilon^*$'
    # labels = [title_label + ':'] + labels
    # handles = [plt.Line2D([], [], linestyle='', label=title_label + ':')] + handles

    fig.legend(handles, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, -0.075))


    plt.tight_layout()
    plt.savefig('plots/hist-da-new.pdf', format='pdf', bbox_inches='tight')
