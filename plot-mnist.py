import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils import *
import pandas as pd
import os, sys

# ---- Font Size Settings ----
plt.rcParams.update({
    'font.size': 15,
    'axes.labelsize': 20,
    'axes.titlesize': 20,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 20,
    'figure.titlesize': 25
})

sns.set_palette('bright')  # or 'deep', 'bright', etc.
sns.set_style('whitegrid')  # or 'darkgrid', 'white', 'dark', etc.


def plot_success_rate(ax, df, eps_comp, clip_norm):
    df_ = df[(df['eps_comp'] == eps_comp) & (df['clip_norm'] == clip_norm)].copy()

    line = sns.lineplot(
        x='beta', y='success_rate', data=df_, ax=ax,
        hue='converged', style='converged', linewidth=2
    )
    ax.set_xscale('log')
    ax.set_xlabel(r'$\beta$')
    ax.get_legend().remove()  # Remove individual legends

if __name__ == '__main__':
    log_fn = sys.argv[1]
    df = pd.read_csv(f'logs/{log_fn}.csv')

    fig, axs = plt.subplots(2, 4, figsize=(16, 6), sharex=True, sharey=True)

    handles = []
    labels = []

    for i, eps_comp in enumerate([.125, .25, .5, 1]):
        for j, clip_norm in enumerate([4, 8]):
            plot_success_rate(axs[j, i], df, eps_comp, clip_norm)
            if i == 0:
                axs[j, i].set_ylabel(f'$C={clip_norm}$')
            if j == 0:
                axs[j, i].set_title(f'$\epsilon^*={eps_comp}$')
    handles, labels = axs[-1, -1].get_legend_handles_labels()

    # Insert the title as the first label
    # title_label = r'Converged'
    # labels = [title_label + ':'] + labels
    # handles = [plt.Line2D([], [], linestyle='', label=title_label + ':')] + handles

    labels = ['Not Converged', 'Converged (Accuracy=98.8%)']

    fig.legend(handles, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, -0.06))


    plt.tight_layout()
    plt.savefig('plots/mnist-da.pdf', format='pdf', bbox_inches='tight')
