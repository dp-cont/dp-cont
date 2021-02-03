import math
from scipy.stats import laplace
from scipy.stats import norm
from percentile_estimator.smooth import Smooth


class SmoothPAK(Smooth):

    def __init__(self, users, args):
        super(SmoothPAK, self).__init__(users, args)

    def obtain_ell(self, p=0):
        if p == 0.99499:
            p = 0.995 * 0.85
            p = 0.99575
        m = self.args.m
        eps = self.args.epsilon
        delta = 1 / self.args.n ** 2
        b = eps / (2 * math.log(1 / delta))
        noise = 'lap'
        beta_lt = 0.3 * 0.02

        ss, p_percentile = self.obtain_ss(m, p, b)

        if noise == 'lap':
            inv_cdf = laplace.ppf(1 - beta_lt)
            a = eps / 2
            ns = laplace.rvs()
        else:
            inv_cdf = norm.ppf(1 - beta_lt)
            a = eps / math.sqrt(-math.log(delta))
            ns = norm.rvs()
        kappa = 1 / (1 - (math.exp(b) - 1) * inv_cdf / a)

        tau = p_percentile + kappa * ss / a * (ns + inv_cdf)
        return max(0, tau)
