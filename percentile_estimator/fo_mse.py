import primitive
from percentile_estimator.fo import FO


class FOMSE(FO):

    def obtain_ell(self, p=0):
        self.hist = primitive.fo(self.m_pdf, self.epsilon)
        thres = self.exact_mse()
        return thres
