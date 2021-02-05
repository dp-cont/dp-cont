import math

import numpy as np
from scipy import optimize

from range_estimator.hierarchy import Hierarchy


class SmoothHierarchy(Hierarchy):

    def __init__(self, users, args):
        Hierarchy.__init__(self, users, args)
        self.fanout = 16
        self.update_fanout()

        self.g = self.args.g
        if self.args.g == 0:
            self.g = self.opt_g()

        self.num_levels = int(math.log(self.n / self.g, self.fanout))
        self.epsilon = self.args.range_epsilon / self.num_levels
        self.granularities = [self.g * self.fanout ** h for h in range(self.num_levels)]

    def opt_g(self):
        def f(x):
            # here x denotes b^s. in the equation in paper (optimizing s).
            # the first part is variance
            # the second part is bias squared
            #  for bias, we assume the bias for each value is theta / 3,
            #  and bias is theta / 3 multiplied by the average number of values in a query.
            #  assuming there are x/2 values in a query, we have average squared bias x^2/36
            # the calculation for the squared average value in a query can be more complicated
            # but we keep it simple as we can only approximate each value's bias to be theta / 3
            return 2 * (self.fanout - 1) * (math.log(self.args.r / x) / math.log(self.fanout)) ** 3 / (self.args.range_epsilon ** 2) \
                   + x ** 2 / 36

        g = int(optimize.fmin(f, 256, disp=False)[0])
        g_exp = math.log(g, self.fanout)
        g_exp = round(g_exp)
        return self.fanout ** g_exp

    def est_precise(self, ell):
        count = self.est_hierarchy(ell)
        count = self.consist(count)
        return count[0]

    def guess(self, ell, hie_leaf, method=None):

        if method == 'naive_smoother':
            u_list = hie_leaf
            return self.set_leaf(ell, u_list, hie_leaf)
        elif method == 'mean_smoother':
            u_list = [np.mean(hie_leaf[:i + 1]) for i in range(len(hie_leaf))]
            return self.set_leaf(ell, u_list, hie_leaf)
        elif method == 'median_smoother':
            u_list = [np.median(hie_leaf[:i + 1]) for i in range(len(hie_leaf))]
            return self.set_leaf(ell, u_list, hie_leaf)
        elif method == 'moving_smoother':
            u_list = [np.mean(hie_leaf[max(0, i - self.args.moving_w):i + 1]) for i in range(len(hie_leaf))]
            return self.set_leaf(ell, u_list, hie_leaf)
        elif method == 'exp_smoother':
            u_list = np.zeros_like(hie_leaf)
            u_list[0] = hie_leaf[0]
            for i in range(1, len(u_list)):
                u_list[i] = u_list[i - 1] * (1 - self.args.exp_smooth_a) + self.args.exp_smooth_a * hie_leaf[i]
            return self.set_leaf(ell, u_list, hie_leaf)
        else:
            raise NotImplementedError(method)

    def set_leaf(self, ell, u_list, hie_leaf):
        leaf_counts = np.zeros(self.n)

        leaf_counts[:self.g] = ell / 2

        for i, u in enumerate(u_list[:-2]):
            leaf_counts[(i + 1) * self.g:(i + 2) * self.g] = u / self.g

        i += 1
        leaf_counts[(i + 1) * self.g:] = u_list[i] / (self.n - (i + 1) * self.g)

        for i, est in enumerate(hie_leaf):
            leaf_counts[min((i + 1) * self.g - 1, self.n - 1)] = est - sum(leaf_counts[i * self.g:(i + 1) * self.g - 1])
        return leaf_counts
