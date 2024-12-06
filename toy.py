import argparse
import math
import pickle
import numpy as np


class Adam:
    def __init__(self, params, betas=(0.9, 0.999)):
        self.params = params
        self.betas = betas
        self.step = 0
        self.exp_avg = np.zeros_like(params)
        self.exp_avg_sq = np.zeros_like(params)

    def update(self, grads, lr):
        self.step += 1
        beta1, beta2 = self.betas
        self.exp_avg *= beta1
        self.exp_avg += (1-beta1) * grads
        self.exp_avg_sq *= beta2
        self.exp_avg_sq += (1-beta2) * grads * grads
        self.params -= lr * self.exp_avg / np.sqrt(self.exp_avg_sq)


class AMSGrad:
    def __init__(self, params, betas=(0.9, 0.999)):
        self.params = params
        self.betas = betas
        self.step = 0
        self.exp_avg = np.zeros_like(params)
        self.exp_avg_sq = np.zeros_like(params)
        self.max_exp_avg_sq = np.zeros_like(params)

    def update(self, grads, lr):
        self.step += 1
        beta1, beta2 = self.betas
        self.exp_avg *= beta1
        self.exp_avg += (1-beta1) * grads
        self.exp_avg_sq *= beta2
        self.exp_avg_sq += (1-beta2) * grads * grads
        self.max_exp_avg_sq = np.maximum(self.max_exp_avg_sq, self.exp_avg_sq)
        self.params -= lr * self.exp_avg / np.sqrt(self.max_exp_avg_sq)


class Adopt:
    def __init__(self, params, betas=(0.9, 0.999)):
        self.params = params
        self.betas = betas
        self.step = 0
        self.exp_avg = np.zeros_like(params)
        self.exp_avg_sq = np.zeros_like(params)

    def update(self, grads, lr):
        if self.step <= 0:
            self.exp_avg_sq += grads * grads
        else:  # self.step > 0
            beta1, beta2 = self.betas
            self.exp_avg *= beta1
            self.exp_avg += (1-beta1) * grads / np.sqrt(self.exp_avg_sq)
            self.params -= lr * self.exp_avg
            self.exp_avg_sq *= beta2
            self.exp_avg_sq += (1-beta2) * grads * grads

        self.step += 1


algorithm_names = ['adam', 'amsgrad', 'adopt']


def optim_algo(args):
    name = args.algo
    betas = (0.9, args.beta2)
    params = np.empty((args.samples,))
    params.fill(args.init)
    if name == 'adam':
        return Adam(params, betas)
    elif name == 'amsgrad':
        return AMSGrad(params, betas)
    elif name == 'adopt':
        return Adopt(params, betas)
    else:
        raise ValueError(f'unknown name: {name}')


def main(args=None):
    parser = argparse.ArgumentParser(description='Toy problem')
    parser.add_argument('--algo', type=str, default='adam', metavar='ALGO',
                        choices=algorithm_names,
                        help='optimization algorithm: ' +
                             ' | '.join(algorithm_names) + ' (default: adam)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='base learning rate (default: 0.01)')
    parser.add_argument('--beta2', type=float, default=0.999, metavar='B2',
                        help="Adam's beta2 parameter (default: 0.999)")
    parser.add_argument('--noise', type=float, default=10, metavar='K',
                        help="constant k controlling the magnitude of gradient noise (default: 10)")
    parser.add_argument('--steps', type=int, default=2 * 10**6, metavar='T',
                        help="number of iterations (default: 2 * 10**6)")
    parser.add_argument('--samples', type=int, default=1000, metavar='N',
                        help="number of samples (default: 1000)")
    parser.add_argument('--init', type=float, default=0, metavar='P',
                        help="initial value of the parameter (default: 0)")
    parser.add_argument('--interval', type=int, default=1000, metavar='I',
                        help="log interval (default: 1000)")
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help="random seed (default: 1)")
    parser.add_argument('--output', type=str, default='output.pickle', metavar='PATH',
                        help="output file path (default: output.pickle)")
    args = parser.parse_args(args)

    print(args)

    optimizer = optim_algo(args)

    k = args.noise
    prob = 1/k
    a = k**2
    b = -k

    rng = np.random.default_rng(args.seed)
    q = np.array([0.25, 0.5, 0.75])
    data = dict(low=[], median=[], high=[], interval=args.interval)

    for i in range(args.steps):
        choices = rng.binomial(1, prob, args.samples)
        grads = np.where(choices == 1, a, b)
        lr = args.lr / math.sqrt(1 + args.lr * i)
        optimizer.update(grads, lr)
        np.clip(optimizer.params, -1, 1, out=optimizer.params)

        if (i+1) % args.interval == 0:
            low, median, high = np.quantile(optimizer.params, q)
            data["low"].append(low)
            data["median"].append(median)
            data["high"].append(high)

    with open(args.output, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
