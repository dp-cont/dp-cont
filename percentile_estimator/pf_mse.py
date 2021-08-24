import primitive
from percentile_estimator.em_mse import EMMSE


class PFMSE(EMMSE):

    def obtain_ell(self, p=0):
        return primitive.pf(self.quality_func(), self.epsilon, sensitivity=1, monotonic=True) * self.args.s
