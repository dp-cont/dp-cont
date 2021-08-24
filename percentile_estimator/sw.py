import numpy as np

import primitive
from percentile_estimator.estimator import Estimator


class SW(Estimator):

    def obtain_ell(self, p=0):
        self.hist = primitive.sw(self.users.data[:self.args.m], 0, self.users.max_ell, self.epsilon)
        thres = self.thres_percentile()
        return thres

    def thres_percentile(self, p=0):
        if p == 0:
            p = 0.95
        data_cdf = np.cumsum(self.hist) / sum(self.hist)
        counter = 0
        while data_cdf[counter] < p:
            counter += 1
        return counter / len(self.bins) * self.users.max_ell
