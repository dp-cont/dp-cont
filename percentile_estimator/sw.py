import primitive
from percentile_estimator.fo import FO


class SW(FO):

    def obtain_ell(self, p=0):
        # if not self.hist:
        self.hist = primitive.sw(self.users.data[:self.args.m], 0, self.users.max_ell, self.epsilon)
        thres = self.thres_percentile()
        return thres
