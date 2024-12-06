import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt


def main(args=None):
    parser = argparse.ArgumentParser(description='Plot for toy problem')
    parser.add_argument('paths', nargs='+', metavar='PATH', help='path to output file of toy problem')
    args = parser.parse_args(args)

    print(args)

    for filename in args.paths:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        y = np.array(data["median"])
        y1 = np.array(data["low"])
        y2 = np.array(data["high"])
        interval = data["interval"] if "interval" in data else 1
        x = range(interval, len(y)*interval+1, interval)
        plt.plot(x, y, label=filename)
        plt.fill_between(x, y1, y2, alpha=.5, linewidth=0)
    plt.ylim(-1.05, 1.05)
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
