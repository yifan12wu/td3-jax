import os

import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import matplotlib.pylab as pylab
params = {'legend.fontsize': 'large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'large',
         'ytick.labelsize':'large'}
pylab.rcParams.update(params)


ENVS = ['HalfCheetah-v3', 'Ant-v3', 'Walker2d-v3', 'Hopper-v3']
N_SEEDS = 5
COLOR = 'royalblue'
EVAL_FREQ = 5000


def plot_fill(x, y, yerr, color=None, alpha_fill=0.1, ax=None, label='', ls='-'):
    ax = ax if ax is not None else plt.gca()
    ymin = y - yerr
    ymax = y + yerr
    handle, = ax.plot(x, y, color=color, lw=2, label=label, ls=ls)
    ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)
    return handle


def main():
    # setup log directories
    output_dir = os.path.join(os.getcwd(), 'plots')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for env in ENVS:
        results = []
        output_file = os.path.join(output_dir, f'TD3_{env}.png')
        for seed in range(N_SEEDS):
            result_file = os.path.join(os.getcwd(), f'results/TD3_{env}_{seed}.txt')
            results.append(np.loadtxt(result_file))
        results = np.array(results)
        y_mean = np.mean(results, axis=0)
        y_std = np.std(results, axis=0)
        x = np.arange(y_mean.shape[0]) * EVAL_FREQ / 1e6
        handle = plot_fill(x, y_mean, y_std, color=COLOR, label='TD3')
        plt.title(env)
        plt.legend(handles=[handle], framealpha=0.0)
        plt.xlabel('training steps (1e6)')
        plt.ylabel('episodic returns')
        plt.savefig(output_file, bbox_inches='tight')
        plt.clf()
        print('Plot saved at {}.'.format(output_file))


if __name__ == '__main__':
    main()