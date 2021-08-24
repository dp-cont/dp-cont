import primitive
from percentile_estimator.sw_mse import SWMSE


class SWSMSE(SWMSE):

    def obtain_ell(self, p=0):
        self.hist = primitive.sw(self.users.data[:self.args.m], 0, self.users.max_ell, self.epsilon, smoothing=True)
        # self.verbose_plot_hist_plain()
        thres = self.exact_mse()
        return thres
