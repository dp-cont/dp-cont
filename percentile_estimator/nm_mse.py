import primitive
from percentile_estimator.em_mse import EMMSE


class NMMSE(EMMSE):

    def obtain_ell(self, p=0):
        return primitive.nm(self.quality_func(), self.epsilon, sensitivity=1, monotonic=True) * self.args.s
