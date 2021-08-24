import primitive
from percentile_estimator.sw import SW


class SWS(SW):

    def obtain_ell(self, p=0):
        self.hist = primitive.sw(self.users.data[:self.args.m], 0, self.users.max_ell, self.epsilon, smoothing=True)
        thres = self.thres_percentile()
        return thres