import numpy as np

import primitive
from percentile_estimator.estimator import Estimator


class Laplace(Estimator):

    def __init__(self, users, args):
        Estimator.__init__(self, users, args)
        sensitivity = 2
        beta = sensitivity / self.epsilon
        self.hist = primitive.laplace(beta, self.users.m_pdf) / self.args.m

    def obtain_ell(self, p=0):
        if p == 0:
            p = 0.95
        data_cdf = np.cumsum(self.hist)
        counter = 0
        while data_cdf[counter] < p:
            counter += 1
        return counter
