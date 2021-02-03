import numpy as np

import primitive
from range_estimator.estimator import Estimator


class HM(Estimator):

    def __init__(self, users, args):
        Estimator.__init__(self, users, args)
        self.epsilon = self.args.epsilon

    def est_precise(self, ell):
        cur_data = np.copy(self.users.data[self.args.m:])
        cur_data[cur_data > ell] = ell
        # original hm runs slow, use an approximation form
        # return primitive.hm(cur_data, 0, ell, self.epsilon)
        if self.epsilon > 1.29:
            return primitive.pm(cur_data, 0, ell, self.epsilon)
        else:
            return primitive.sr(cur_data, 0, ell, self.epsilon)
