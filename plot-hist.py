import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils import *
import pandas as pd

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


def plot_success_rate(ax, df, num_bins):
    df_ = df[df['num_bins'] == num_bins]

    line = sns.lineplot(
        x='beta', y='success_rate', data=df_, ax=ax,
        hue='eps_comp', style='eps_comp', linewidth=2
    )
    ax.set_xscale('log')
    ax.set_xlabel(r'$\beta$')
    ax.set_ylabel('Success Rate')
    ax.get_legend().remove()  # Remove individual legends

def plot_l2(ax, df, num_bins):
    df_ = df[df['num_bins'] == num_bins].copy()
    df_['l2_theoretical'] = np.sqrt(df_['num_bins']) * df_['sigma']

    sns.lineplot(
        x='eps_comp', y='l2_theoretical', data=df_, ax=ax,
        linestyle='--', linewidth=2, label='Theoretical'
    )
    sns.lineplot(
        x='eps_comp', y='l2', data=df_, ax=ax,
        linestyle='-', linewidth=2, label='Actual'
    )
    ax.set_xscale('log')
    ax.set_xlabel(r'$\epsilon^*$')
    ax.set_ylabel('L2 Error')
    ax.get_legend().remove()  # Remove individual legends

if __name__ == '__main__':
    df = pd.read_csv('logs/hist-da.csv')

    fig, axs = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey='row')

    handles = []
    labels = []

    for i, num_bins in enumerate([256, 4096, 65536]):
        plot_success_rate(axs[i], df, num_bins)

    handles, labels = axs[-1].get_legend_handles_labels()

    # Insert the title as the first label
    title_label = r'$\epsilon^*$'
    labels = [title_label + ':'] + labels
    handles = [plt.Line2D([], [], linestyle='', label=title_label + ':')] + handles

    fig.legend(handles, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, -0.1))


    plt.tight_layout()
    plt.savefig('plots/hist-da.pdf', format='pdf', bbox_inches='tight')
