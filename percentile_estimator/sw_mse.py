import primitive
from percentile_estimator.fo import FO


class SWMSE(FO):

    def obtain_ell(self, p=0):
        self.hist = primitive.sw(self.users.data[:self.args.m], 0, self.users.max_ell, self.epsilon)
        # self.verbose_plot_hist_plain()
        thres = self.exact_mse(postprocess=True)
        # print(thres)
        return thres
