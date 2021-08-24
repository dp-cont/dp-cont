import numpy as np
from scipy.optimize import curve_fit

import primitive
from percentile_estimator.estimator import Estimator


class FO(Estimator):

    def __init__(self, users, args):
        Estimator.__init__(self, users, args)
        self.m_pdf, self.bins = np.histogram(
            self.users.data[:self.args.m],
            bins=np.linspace(0, self.users.max_ell + 1, int((self.users.max_ell + 2) / self.args.s))
        )
        self.bins = self.bins[:-1]
        self.hist = None

    def obtain_ell(self, p=0):
        self.hist = primitive.fo(self.m_pdf, self.epsilon)
        thres = self.thres_percentile(p)
        # thres = self.approx_mse()
        # thres = self.exact_mse()
        return thres

    def thres_percentile(self, p=0):
        if p == 0:
            p = 0.95
        data_cdf = np.cumsum(self.hist) / sum(self.hist)
        counter = 0
        while data_cdf[counter] < p:
            counter += 1
        return counter / len(self.bins) * self.users.max_ell

    def approx_mse(self):
        ee = np.exp(self.args.range_epsilon)
        n = (self.users.n - self.users.m)
        noise_std = (ee + 1) / (ee - 1) * n ** 0.5

        data_cdf = np.cumsum(self.hist) / sum(self.hist) * n
        bias = n - data_cdf * n
        self.bins = np.linspace(0, self.users.max_ell, len(bias))

        return np.argmin(bias + noise_std * self.bins) / len(self.bins) * self.users.max_ell

    def exact_mse(self, postprocess=False):
        ee = np.exp(self.args.range_epsilon)
        n = (self.users.n - self.users.m)
        noise_std = (ee + 1) / (ee - 1)
        if self.args.range_epsilon > 0.61:
            # equation 8 of the icde 2019 paper
            eesqrt = np.exp(self.args.range_epsilon / 2)
            noise_var1 = (eesqrt + 3) / (3 * eesqrt * (eesqrt - 1))
            noise_var2 = (ee + 1) ** 2 / (eesqrt * (ee - 1) ** 2)
            noise_std = (noise_var1 + noise_var2)

        self.bins = np.linspace(0, self.users.max_ell, len(self.hist))
        self.hist = self.hist / sum(self.hist)

        if postprocess:
            # self.my_fit()
            self.my_remove_zero()

        # self.hist = self.hist / sum(self.hist) * n
        mse = []
        expected_range = n / 5
        # expected_range = 1
        for theta_i, theta in enumerate(self.bins):
            noise = (noise_std * theta) ** 2 * expected_range
            bias_sq = (expected_range * sum(self.hist[theta_i:] * (self.bins[theta_i:] - theta))) ** 2
            mse.append((noise + bias_sq) ** 0.5)
        mse = np.array(mse)
        return np.argmin(mse) / len(self.bins) * self.users.max_ell

    def my_fit(self):
        max_index = np.argmin(self.hist)

        def func(x, a, b):
            return a * np.exp(-b * x)
        popt, _ = curve_fit(func, self.bins[max_index + 2:], self.hist[max_index + 2:] / 256, p0=[1000, 1], maxfev=5000)
        self.hist = popt[0] * np.exp(-popt[1] * self.bins / 256)
        self.hist[self.hist < 1] = 0

    def my_remove_zero(self):
        def zero_runs(a):
            # Create an array that is 1 where a is 0, and pad each end with an extra 0.
            iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
            absdiff = np.abs(np.diff(iszero))
            # Runs start and end where absdiff is 1.
            ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
            return ranges

        self.hist[self.hist < sum(self.hist) / 1000] = 0
        runs = zero_runs(self.hist)
        truncated = False
        for run in runs:
            if len(run) >= 5:
                self.hist[run[0]:] = 0
                truncated = True
                break
        if not truncated:
            self.hist[runs[-1][0]:] = 0

        mask = self.hist > 0
        diff = (1 - sum(self.hist)) / sum(mask)
        self.hist[mask] += diff

    def verbose_plot_hist_plain(self):

        import matplotlib.pyplot as plt
        from matplotlib.pyplot import figure
        plt.figure()
        plt.style.use('seaborn-darkgrid')
        w, h = 6.4, 3.6
        figure(figsize=(w, h))
        plt.rcParams["font.size"] = 18
        plt.yticks(fontsize=18)
        # plt.yscale("log")
        # plt.hist(self.data, density=True, bins=200, range=(0, max(self.data)))
        bins = np.linspace(0, self.users.max_ell, 200)

        sw_data = np.zeros(self.args.m)
        old_counter = 0
        for i, est in enumerate(self.hist):
            sw_data[old_counter:old_counter + int(est)] = i / len(self.hist) * self.users.max_ell
            old_counter += int(est)

        plt.hist(self.users.data[:self.args.m], density=True, bins=bins, alpha=0.5, label='true')
        plt.hist(sw_data, density=True, bins=bins, alpha=0.5, label='sw')
        plt.show()
        dist_name = self.args.user_type
        # plt.savefig('results/dist/' + dist_name + "_est_eps%d.pdf" % self.epsilon, bbox_inches='tight')
        plt.cla()
