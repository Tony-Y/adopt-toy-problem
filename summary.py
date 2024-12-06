import argparse
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


def plot_panel(k, algo, filenames, ax):
    for label, filename in filenames.items():
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        y = np.array(data["median"])
        y1 = np.array(data["low"])
        y2 = np.array(data["high"])
        interval = data["interval"]
        x = range(interval, len(y)*interval+1, interval)
        ax.plot(x, y, label=r'$\beta_{2}='+label+r'$')
        ax.fill_between(x, y1, y2, alpha=.5, linewidth=0)
    ax.set_ylim(-1.05, 1.05)
    ax.legend()
    ax.set_title(r'$k='+str(k)+r'$'+f', {algo}')
    ax.set_xlabel('Step')
    ax.set_ylabel(r'$\theta$')
    ax.grid()


def main(args=None):
    parser = argparse.ArgumentParser(description='Summary of toy problem')
    parser.add_argument('path', default='data', metavar='PATH',
                        help='path to output directory of toy problem (default: data)')
    args = parser.parse_args(args)

    print(args)

    fig, axs = plt.subplots(2, 3, figsize=(12, 6), layout="constrained")

    for i, k in enumerate([10, 50]):
        for j, algo in enumerate(['adam', 'amsgrad', 'adopt']):
            filenames = {
                '0.1': os.path.join(args.path, f'samp_k{k}_b0-1_{algo}.pickle'),
                '0.5': os.path.join(args.path, f'samp_k{k}_b0-5_{algo}.pickle'),
                '0.999': os.path.join(args.path, f'samp_k{k}_{algo}.pickle'),
            }
            plot_panel(k, algo, filenames, axs[i, j])

    plt.show()


if __name__ == '__main__':
    main()
