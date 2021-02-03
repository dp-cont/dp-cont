import math

import pickle
from os import path
import numpy as np
from scipy.stats import laplace
from scipy.stats import norm

import primitive
from percentile_estimator.estimator import Estimator


class Smooth(Estimator):

    def __init__(self, users, args):
        super(Smooth, self).__init__(users, args)

    def obtain_ss(self, m, p, b):
        user_file_name = 'data/ss/%s-%d-%f-%f' % (self.args.user_type, m, p, b)
        if not path.exists(user_file_name):
            sorted_data = np.sort(np.copy(self.users.data[:m]))
            p_rank = int(self.args.m * p)
            p_percentile = sorted_data[p_rank]
            ss = primitive.smooth_ell(p_rank, b, self.users.max_ell, sorted_data)
            pickle.dump([ss, p_percentile], open(user_file_name, 'wb'))

        [ss, p_percentile] = pickle.load(open(user_file_name, 'rb'))
        return ss, p_percentile

    def obtain_ell(self, p=0):
        # 0.9499 is the default p for other methods,
        if p == 0.99499:
            p = 0.995 * 0.85
            p = 0.995
        noise = 'lap'
        m = self.args.m
        n = self.args.n
        eps = self.args.epsilon
        delta = 1 / n ** 2
        # delta = 2e-20
        b = eps / (2 * math.log(1 / delta))

        ss, p_percentile = self.obtain_ss(m, p, b)

        if noise == 'lap':
            a = eps / 2
        else:
            a = eps / math.sqrt(-math.log(delta))

        if noise == 'lap':
            ns = laplace.rvs()
        else:
            ns = norm.rvs()

        return max(0, p_percentile + ss / a * ns)
