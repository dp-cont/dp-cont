import numpy as np

import primitive
from percentile_estimator.estimator import Estimator


class SWMSE(Estimator):

    def obtain_ell(self, p=0):
        self.hist = primitive.sw(self.users.data[:self.args.m], 0, self.users.max_ell, self.epsilon)
        thres = self.exact_mse(postprocess=True)
        return thres

    def exact_mse(self, postprocess=False):
        ee = np.exp(self.args.range_epsilon)
        n = (self.users.n - self.users.m)

        # equation 8 of the icde 2019 paper (worst case error for both cases)
        if self.args.range_epsilon > 0.61:
            eesqrt = np.exp(self.args.range_epsilon / 2)
            noise_var1 = (eesqrt + 3) / (3 * eesqrt * (eesqrt - 1))
            noise_var2 = (ee + 1) ** 2 / (eesqrt * (ee - 1) ** 2)
            noise_std = (noise_var1 + noise_var2) ** 0.5
        else:
            noise_std = (ee + 1) / (ee - 1)

        self.bins = np.linspace(0, self.users.max_ell, len(self.hist))
        self.hist = self.hist / sum(self.hist)

        if postprocess:
            self.remove_trailing_zero()

        mse = []
        for theta_i, theta in enumerate(self.bins):
            noise = (noise_std * theta) ** 2 * n / 3
            bias_sq = n ** 2 / 24 * ((sum(self.hist[theta_i:] * (self.bins[theta_i:] - theta))) ** 2)
            mse.append(noise + bias_sq)
        mse = np.array(mse)
        return np.argmin(mse) / len(self.bins) * self.users.max_ell

    def remove_trailing_zero(self):
        def zero_runs(a):
            # Create an array that is 1 where a is 0, and pad each end with an extra 0.
            iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
            absdiff = np.abs(np.diff(iszero))
            # Runs start and end where absdiff is 1.
            ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
            return ranges

        self.hist[self.hist < sum(self.hist) / 10000] = 0
        runs = zero_runs(self.hist)
        # print(runs)
        for run in runs:
            if run[1] - run[0] >= 5:
                self.hist[run[0]:] = 0
                # print(run[0])
                break

        mask = self.hist > 0
        diff = (1 - sum(self.hist)) / sum(mask)
        self.hist[mask] += diff
