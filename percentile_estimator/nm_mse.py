import math

import numpy as np

import primitive
from percentile_estimator.estimator import Estimator


class NMMSE(Estimator):

    def obtain_ell(self, p=0):
        return primitive.nm(self.quality_func(), self.epsilon, sensitivity=1, monotonic=True) * self.args.s

    def quality_func(self):
        m = self.args.m
        data = self.users.data[:m]
        max_ell = self.users.max_ell

        num_levels = int(math.log(self.users.n - self.users.m, 16)) + 1

        c = 60
        noise_var = 2 * (16 - 1) * num_levels ** 3 / self.args.range_epsilon ** 2
        noise_std = noise_var ** 0.5 * 3 * self.users.m / (self.users.n - self.users.m) / c

        n = len(data)

        n_ell, bins = np.histogram(data, bins=np.linspace(0, max_ell + 1, int((max_ell + 2) / self.args.s)))
        n_ell = np.cumsum(n_ell)
        counts = n_ell - n

        # self.my_plot(data, max_ell, num_levels, noise_std, counts)

        return counts - noise_std * bins[:-1]

    def my_plot(self, data, max_ell, num_levels, noise_std, counts):
        m_ell, bins = np.histogram(data, bins=np.linspace(0, max_ell + 1, int((max_ell + 2) / self.args.s)))
        true_mse = []
        for theta in range(len(m_ell)):
            noise = 2 * (16 - 1) * num_levels ** 3 * theta ** 2 / self.args.range_epsilon ** 2
            bias_sq = ((self.users.n - self.users.m) / 3 / self.users.m * sum(
                m_ell[theta:] * (bins[theta:-1] - theta))) ** 2
            true_mse.append(math.sqrt(noise + bias_sq))
        true_mse = np.array(true_mse)

        dist_name = self.args.user_type
        plot_name = '%s_m%s_noise' % (dist_name, self.args.m)
        self.verbose_plot_quality(- noise_std * bins[:-1], counts, true_mse, 'results/dist/em_mse/', plot_name)
