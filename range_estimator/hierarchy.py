import math

import numpy as np

import primitive
from estimator.estimator import Estimator


class Hierarchy(Estimator):

    def __init__(self, users, args):
        Estimator.__init__(self, users, args)
        self.fanout = self.args.hie_fanout
        self.n = self.users.n - self.users.m
        self.num_levels = int(math.log(self.n, self.fanout))
        self.epsilon = self.args.range_epsilon / self.num_levels
        self.granularities = [self.fanout ** h for h in range(self.num_levels)]

    def update_fanout(self):
        self.num_levels = int(math.log(self.n, self.fanout))
        self.epsilon = self.args.range_epsilon / self.num_levels
        self.granularities = [self.fanout ** h for h in range(self.num_levels)]

    def est_hierarchy(self, ell):
        cur_data = np.copy(self.users.data[self.args.m:])
        cur_data[cur_data > ell] = ell

        count = []
        for granularity in self.granularities:
            num_slots = np.ceil(self.n / granularity).astype(int)
            count_l = np.zeros(num_slots)
            for slot in range(num_slots):
                count_l[slot] = sum(cur_data[slot * granularity: (slot + 1) * granularity])
            count.append(count_l)

        for h in range(self.num_levels):
            count[h] = primitive.laplace(ell / self.epsilon, count[h])

        return count

    def consist(self, count):
        # requires a complete tree

        fanout = self.fanout

        # leaf to root
        for h in range(1, len(count)):
            coeff = fanout ** (h + 1)
            coeff2 = fanout ** h

            for est_i in range(len(count[h])):
                children_est = sum(count[h - 1][est_i * fanout: (est_i + 1) * fanout])
                count[h][est_i] = (coeff - coeff2) / (coeff - 1) * count[h][est_i] + (coeff2 - 1) / (
                        coeff - 1) * children_est

        # root to leaf
        for h in range(len(count) - 1, 0, -1):
            for est_i in range(len(count[h])):
                children_est = sum(count[h - 1][est_i * fanout: (est_i + 1) * fanout])

                diff = (count[h][est_i] - children_est) / fanout
                count[h - 1][est_i * fanout: (est_i + 1) * fanout] += diff

        return count
