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
    'font.size': 12,
    'axes.labelsize': 15,
    'axes.titlesize': 15,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 15,
    'figure.titlesize': 20,
    'mathtext.fontset': 'stix',
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
})




def plot_success_rate(ax, df, eps_comp, clip_norm):
    df_ = df[(df['eps_comp'] == eps_comp) & (df['clip_norm'] == clip_norm)].copy()

    line = sns.lineplot(
        x='beta', y='success_rate', data=df_, ax=ax,
        hue='hue', style='hue', linewidth=1.5
    )
    ax.set_xscale('log')
    ax.set_xlabel(r'$\beta$')
    ax_legend = ax.get_legend()
    if ax_legend is not None:
        ax_legend.remove()
    # ax.get_legend().remove()  # Remove individual legends

if __name__ == '__main__':
    log_fn = sys.argv[1]
    df = pd.read_csv(f'logs/{log_fn}.csv')

    fig, axs = plt.subplots(2, 4, figsize=(16, 6), sharex=True, sharey=True)

    # create a hue manually, combining converged and gamma
    # format (latex): Converged ($\gamma=...$) or Not Converged ($\gamma=...$)
    # gamma: retrieve from the dataframe, keep the most signficnat digit, and format in scientific notation if necessary

    # convert gamma to aeb, where a is one digit, e is 'e', b is integer
    df['gamma_a'] = df['gamma'].apply(lambda x: float(f"{x:.1e}".split('e')[0]))
    df['gamma_b'] = df['gamma'].apply(lambda x: int(f"{x:.1e}".split('e')[1]))
    
    
    df['hue'] = df.apply(lambda row: f'Converged ($\\gamma={row["gamma_a"]}\\times 10^{{{row["gamma_b"]}}}$)' if row['converged'] else f'Not Converged ($\\gamma={row["gamma_a"]}\\times 10^{{{row["gamma_b"]}}}$)', axis=1)

    # print(df)

    # # sort the dataframe by hue, not converged first, then converged
    # df['hue_order'] = df['hue'].apply(lambda x: 0 if 'Not Converged' in x else 1)
    # df = df.sort_values(by='hue_order')
    # df = df.drop(columns=['hue_order', 'gamma_a', 'gamma_b'])


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

    # change the order of handels and labels
    new_handles = []
    new_labels = []
    for k in range((len(handles) + 1)//2):
        if 2*k + 1 < len(handles):
            new_handles.append(handles[2*k + 1])
            new_labels.append(labels[2*k + 1])
        if 2*k < len(handles):
            new_handles.append(handles[2*k])
            new_labels.append(labels[2*k])
        
    handles = new_handles
    labels = new_labels
    # print(labels)
    fig.legend(handles, labels, loc='lower center', ncol=(len(labels) + 1)//2, bbox_to_anchor=(0.5, -0.12))
    


    plt.tight_layout()
    plt.savefig('plots/mnist-da-new.pdf', format='pdf', bbox_inches='tight')