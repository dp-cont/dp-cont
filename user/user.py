import abc

import numpy as np


class User(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, args):
        self.args = args

        self.data = np.array([])
        self.up_groups = np.array([])
        self.up_groups_dist = []
        self.m_pdf = []
        self.m = self.args.m
        self.n = 0
        self.num_up_group = 0
        self.max_ell = 0

        self.initial_generate()
        self.synopsis_generate()

        # self.verbose_print_hist()
        # self.verbose_plot_hist()
        # self.verbose_plot_hist_plain()
        # self.verbose_plot_thres()
        # self.verbose_print_thres()

    def synopsis_generate(self):

        while self.n < self.args.n:
            self.data = np.append(self.data, self.data[self.args.n - self.n:0:-1])
            self.n = len(self.data)
        if self.n > self.args.n:
            self.data = self.data[:self.args.n]
        self.n = self.args.n

        self.m_pdf, _ = np.histogram(self.data[:self.m], bins=range(self.max_ell + 1))

        if self.args.metric == 'up_ell':
            self.num_up_group = int((self.n - self.m) / self.args.n_up_group)
            self.num_up_group = 12
            # omit the first m and final dividant
            self.up_groups = np.split(self.data[self.m: self.m + self.num_up_group * self.args.n_up_group],
                                      self.num_up_group)
            for group_i in range(self.num_up_group):
                n_pdf, _ = np.histogram(self.up_groups[group_i], bins=range(self.max_ell))
                self.up_groups_dist.append(n_pdf)

    def verbose_print_thres(self):
        true_thres = []
        for group_i in range(self.num_up_group):
            n_cdf = np.cumsum(self.up_groups_dist[group_i])
            thres = np.searchsorted(n_cdf, self.args.p * 100, side='left')
            print(thres)
            true_thres.append(thres)

    @abc.abstractmethod
    def initial_generate(self):
        return

    def verbose_print_hist(self):
        print(sum(self.data))

        sorted_dist = np.copy(self.data)
        sorted_dist = -np.sort(-sorted_dist)
        print(sorted_dist[:20])

    def verbose_plot_hist(self):

        sorted_dist = np.copy(self.data)
        sorted_dist = -np.sort(-sorted_dist)

        import matplotlib.pyplot as plt
        from matplotlib.pyplot import figure
        plt.figure()
        plt.style.use('seaborn-darkgrid')
        w, h = 6.4, 3.6
        figure(figsize=(w, h))
        plt.rcParams["font.size"] = 18
        plt.yticks(fontsize=18)
        # plt.yscale("log")
        plt.plot(sorted_dist)
        dist_name = self.args.user_type
        if dist_name == 'synthesize':
            dist_name = self.args.dist
        # plt.show()
        plt.savefig('results/dist/' + dist_name + ".pdf", bbox_inches='tight')
        plt.cla()

    def verbose_plot_hist_plain(self):

        import matplotlib.pyplot as plt
        from matplotlib.pyplot import figure
        plt.figure()
        plt.style.use('seaborn-darkgrid')
        w, h = 6.4, 3.6
        figure(figsize=(w, h))
        plt.rcParams["font.size"] = 18
        plt.yticks(fontsize=18)
        # plt.yscale("log")
        plt.hist(self.data, density=True, bins=200, range=(0, max(self.data)))
        dist_name = self.args.user_type
        if dist_name == 'synthesize':
            dist_name = self.args.dist
        plt.show()
        # plt.savefig('results/dist/' + dist_name + "_plain.pdf", bbox_inches='tight')
        plt.cla()

    def verbose_plot_thres(self):

        tmp = np.copy(self.data)
        num_point = 100
        tmps = np.array_split(tmp, num_point)

        percentiles = [85, 95, 99.5, 99.95]
        thres = np.zeros((len(percentiles), num_point))
        for percentile_i, percentile in enumerate(percentiles):
            for i in range(num_point):
                thres[percentile_i][i] = np.percentile(tmps[i], percentile)

        import matplotlib.pyplot as plt
        from matplotlib.pyplot import figure
        plt.figure()
        plt.style.use('seaborn-darkgrid')
        w, h = 6.4, 3.6
        figure(figsize=(w, h))
        plt.rcParams["font.size"] = 18
        plt.yticks(fontsize=18)
        plt.yscale("log")
        for percentile_i, percentile in enumerate(percentiles):
            plt.plot(thres[percentile_i])
            dist_name = self.args.user_type
            if dist_name == 'synthesize':
                dist_name = self.args.dist
            plt.savefig('results/dist/' + dist_name + '_thres_' + str(percentile) + '.pdf', bbox_inches='tight')
            plt.cla()
