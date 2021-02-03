import numpy as np
from percentile_estimator.estimator import Estimator


class NPP(Estimator):

    # noise free version to get percentile
    def obtain_ell(self, p=0.95):
        if self.p > 0:
            p = self.p / 100
        q_ans = self.quality_func(p)

        return np.argmax(q_ans > 0) * self.args.s

    def quality_func(self, p):
        data_cdf = np.cumsum(self.users.m_pdf)
        n_hat = p * data_cdf[-1]
        q_ans = data_cdf - n_hat
        q_ans = q_ans[::self.args.s] if self.args.s > 1 else np.repeat(q_ans, int(1 / self.args.s))
        return q_ans
