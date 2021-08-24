import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure

from figures.draw import Plot


class PlotScript(Plot):
    def __init__(self):
        Plot.__init__(self)

        self.methods = {
            'all': [
                'svt_hie',
                'ss_hie',
                'svt_bt',
                'pak',
            ],
            'ell': [
                'svt',
                'nm_mse',
                'nm',
                'np_p',
                'np_mse',
                'baseline',
                'smooth',
                'smooth_pak',
            ],
            'range': [
                'basic_hierarchy',
                'opt_fanout_hierarchy',
                'consist_hierarchy',
                'guess_hierarchy',
            ],
            'smooth': [
                'naive_smoother',
                'mean_smoother',
                'median_smoother',
                'moving_smoother',
                'exp_smoother',
            ]
        }
        self.method_names = {
            'svt_hie': 'ToPS',
            'svt_bt': 'NM-E',
            'ss_hie': '$\\hat{H}^c_{16}$',
            'pak': 'PAK',
            'sw_hm': 'ToPL',
            'truth': 'True',

            'svt': 'SVT-P',
            'nm_mse': 'NM-E',
            'nm': 'EM-P',
            'np_p': 'NP-P',
            'np_p85': '85',
            'np_p90': '90',
            'np_p95': '95',
            'np_p99.5': '99.5',
            'np_p99.9': '99.9',
            'np_mse': 'NP-E',
            'baseline': 'Base',
            'smooth': 'S-P',
            'smooth_pak': 'S-PAK',
            'fo': 'FO',
            'fo_mse': 'FO-E',
            'sw': 'SW',
            'sws': 'SWS',
            'sw_mse': 'SW-W',
            'sws_mse': 'SW',

            'naive_smoother': 'Recent',
            'mean_smoother': 'Mean',
            'median_smoother': 'Median',
            'moving_smoother': 'Move',
            'exp_smoother': 'ExpS',

            'basic_hierarchy': '$H_2$',
            'opt_fanout_hierarchy': '$H_{16}$',
            'consist_hierarchy': '$H^c_{16}$',
            'guess_hierarchy': '$\\hat{H}^c_{16}$',
        }
        self.method_indexs = {
            'em_mse': 0,
            'nm_mse': 0,
            'pf_mse': 5,
            'smooth': 1,
            'smooth_pak': 2,
            'svt': 3,
            'nm': 4,

            'np_p': 5,
            'np_p85': 1,
            'np_p90': 2,
            'np_p95': 3,
            'np_p99.5': 4,
            'np_p99.9': 5,
            'np_mse': 6,
            'baseline': 7,
            'fo': 1,
            'fo_mse': 2,
            'sw': 3,
            'sws': 4,
            'sw_mse': 5,
            'sws_mse': 6,

            'naive_smoother': 0,
            'mean_smoother': 1,
            'median_smoother': 2,
            'moving_smoother': 3,
            'exp_smoother': 4,

            'basic_hierarchy': 0,
            'opt_fanout_hierarchy': 1,
            'consist_hierarchy': 2,
            'guess_hierarchy': 3,

            'svt_hie': 0,
            'svt_bt': 2,
            'ss_hie': 3,
            'pak': 1,
            'truth': 4,
            'sw_hm': 5,
        }
        self.ns = {
            'time': 8735777,
            'fare': 8704495,
            'dist': 8704477,
            'kosarak': 990002,
            'pos': 515597,
            'web1': 2000000,
            'web2': 2000000,
            'retail': 2000000,
            'star': 2000000,
            'aol': 2000000,
            'dns_1k': 1141961,
            'google_1k': 355872,
        }
        self.user_types = []

        # unit: time is in second, fare is in cents, dist is in 0.01 mile, transactions are number of items
        self.max_ell_dict = {'time': 21600, 'dist': 20000, 'fare': 30000,
                             'kosarak': 42178, 'pos': 1638, 'aol': 2290685,
                             'retail': 16470, 'web1': 497, 'web2': 3340,
                             'star': 2088, 'dns_1k': 2000, 'google_1k': 500}

    # empirical evaluation

    def eps(self, vary, metric, m):
        ylabel = "MSE"
        if 'range' in metric:
            methods = [
                'basic_hierarchy',
                'opt_fanout_hierarchy',
                'consist_hierarchy',
                'guess_hierarchy',
                'baseline'
            ]
        elif 'ell' in metric:
            methods = [
                'nm_mse',
                # 'em_mse',
                # 'pf_mse',
                'smooth',
                'smooth_pak',
                'baseline',
            ]
            if vary == 'eps_small':
                methods = methods[:-3]
            if vary in ['eps_large', 'eps_ldp']:
                methods = [
                    # 'sw',
                    'sw_mse',
                    # 'sws',
                    'sws_mse',
                    'baseline',
                ]
            if vary in ['eps_range_medium', 'eps_range_small']:
                methods = ['np_p85', 'np_p90', 'np_p95', 'np_p99.5', 'np_p99.9',
                           'em_mse'
                           ]
        elif 'smooth' in metric:
            methods = [
                'naive_smoother',
                'mean_smoother',
                'median_smoother',
                'moving_smoother',
                'exp_smoother',
            ]
        elif metric in ['query_mse', 'query_mae', 'query_mmae', 'query_mmse']:
            # methods = self.methods['all']
            methods = [
                'svt_hie',
                'ss_hie',
                'svt_bt',
                'pak',
            ]
            ylabel = metric[6:].upper()

        my_ncol = 4 if len(methods) == 4 else 3

        for user_type in self.user_types:
            # file_name = fig_name = user_type + '_m' + str(m) + '_' + metric
            file_name = fig_name = user_type + '_m' + str(m) + '_' + metric
            print(file_name)
            path = vary + '/' + file_name
            json_data, pars = self.load_data(path, vary)
            div = 1

            for method_index, method in enumerate(methods):
                method_index = self.method_indexs[method]
                method_name = self.method_names[method]

                x_data, y_data, y_error = self.read_value(json_data, pars, method, div)
                print(method, y_data)

                self.plot_errorbar(x_data, y_data, y_error, method_name, method_index)

            plt.legend(fontsize=self.legend_size, ncol=my_ncol, loc=(0, 1.02))
            plt.yscale("log")
            # plt.xlim([0.2, 2])
            plt.xlabel("$\\epsilon$", fontsize=self.xlabel_font_size)
            plt.ylabel(ylabel=ylabel, fontsize=self.xlabel_font_size)
            self.draw_fig(vary + '/', fig_name)

    def ell_p(self, metric, eps, range_eps):
        m = 65536
        methods = ['svt',
                   'nm',
                   'smooth',
                   'smooth_pak',
                   'np_p',
                   'baseline'
                   ]
        for user_type in self.user_types:
            file_name = fig_name = user_type + '_m' + str(m) + '_eps' + str(int(eps * 10)) + '_' + str(
                int(range_eps * 10)) + '_' + metric
            print(file_name)
            path = 'p/' + file_name
            json_data, pars = self.load_data(path, 'p')
            n = self.ns[user_type]
            # div = (n ** 2)
            div = 1
            method_min = {}
            method_max = {}

            for method_index, method in enumerate(methods):
                method_index = self.method_indexs[method]
                method_name = self.method_names[method]

                x_data, y_data, y_error = self.read_value(json_data, pars, method, div)
                method_max[method] = max(y_data)
                method_min[method] = min(y_data)

                self.plot_errorbar(x_data, y_data, y_error, method_name, method_index)

            plt.legend(fontsize=self.legend_size, ncol=3, loc=(0, 1.02))
            plt.yscale("log")
            # plt.xlim([0.8, 0.995])
            plt.ylim([method_min['np_p'] / 2, method_max['baseline'] * 2])
            plt.xlabel("$p$", fontsize=self.xlabel_font_size)
            plt.ylabel("MSE", fontsize=self.xlabel_font_size)
            self.draw_fig('p/', fig_name)

    def precise(self, vary, m):
        if vary in ['eps_ldp', 'eps_large']:
            methods = [
                'sw_hm',
            ]
        else:
            methods = [
                'svt_hie',
                'pak',
            ]
        my_ncol = 2
        metric = 'precise'
        figure(figsize=(12, 4))
        for user_type in self.user_types:
            file_name = user_type + '_m' + str(m) + '_' + metric
            print(file_name)
            path = vary + '/' + file_name
            json_data, pars = self.load_data(path, vary)
            for comp_method in methods:
                for par in pars:
                    fig_name = file_name + '_' + comp_method + '_' + str(int(float(par) * 1000))
                    for method_index, method in enumerate([comp_method, 'truth']):
                        method_index = self.method_indexs[method]
                        method_name = self.method_names[method]

                        y_data = json_data[str(par)][method][0]
                        x_data = range(len(y_data))

                        self.plot_errorbar(x_data, y_data, np.zeros_like(x_data), method_name, method_index)

                    plt.legend(fontsize=self.legend_size, ncol=my_ncol, loc='upper right')
                    plt.ylim([-5, 85])
                    plt.xlabel("Index", fontsize=self.xlabel_font_size)
                    plt.ylabel("Mean", fontsize=self.xlabel_font_size)
                    self.draw_fig(vary + '/', fig_name)

    def dist(self):
        # plt.figure()
        # plt.style.use('seaborn-darkgrid')
        # w, h = 6.4, 3.2
        # figure(figsize=(w, h))
        # plt.rcParams["font.size"] = 18
        # plt.yticks(fontsize=18)

        for user_type in self.user_types:
            if user_type in ['time', 'fare', 'dist']:
                user_file_name = 'data/nyctaxi/taxi201801_'
                data = pickle.load(open(user_file_name + user_type + '.pkl', 'rb'))
            elif user_type in ['kosarak', 'pos', 'web1', 'web2', 'aol', 'retail', 'star']:
                user_file_name = 'data/transaction/'
                data = pickle.load(open(user_file_name + user_type + '_length.pkl', 'rb'))

            # 1. data cleaning: max time is 6h, max fare is $300, max distance is 100 miles
            # data = data[data > 0]
            # data = data[data < self.max_ell_dict[type]]
            # pickle.dump(data, open(user_file_name + user_type + '.pkl', 'wb'))

            # 2. data pre-processing: output the percentiles
            # for m in [50000, -1]:
            #     x_values = np.linspace(0, np.percentile(data[:m], 99.95), 100)
            #     freq, _ = np.histogram(data, x_values)
            #     freq_cum = np.cumsum(freq)
            #     freq_cum = freq_cum / data.size
            #     filename = 'results/dist/' + user_type
            #     if not m == -1:
            #         filename += '_m%d' % m
            #     pickle.dump(x_values, open(filename + '_bin.pkl', 'wb'))
            #     pickle.dump(freq_cum, open(filename + '_cdf.pkl', 'wb'))

            # 3. plot
            # x_values = np.linspace(0, np.percentile(data, 99.5), 20)
            # freq, _ = np.histogram(data, x_values)
            # freq_cum = np.cumsum(freq)
            # freq_cum = freq_cum / data.size
            # plt.plot(x_values[:-1].astype(int), freq_cum)
            # plt.savefig(self.folder + 'dist/' + user_type + "_cdf.pdf", bbox_inches='tight')
            # plt.cla()

            # 4. plot
            # sorted_dist = np.copy(data)
            # sorted_dist = -np.sort(-sorted_dist)
            # print('%s & %d & %d & %d & %d & %d & %d & %.1f \\\\ \\hline' % (user_type, data.size, self.max_ell_dict[user_type], max(data), np.percentile(data, 85), np.percentile(data, 95), np.percentile(data, 99.5), np.mean(data)))
            # plt.yscale("log")
            # plt.plot(sorted_dist)
            # plt.savefig(self.folder + 'dist/' + user_type + ".pdf", bbox_inches='tight')
            # plt.cla()


if __name__ == "__main__":
    plot = PlotScript()

    # ToPS visualization
    # plot.user_types = ['dns_1k']
    # plot.precise('eps_medium', 65536)
    # plot.precise('eps', 65536)

    # Overall performance comparison
    # plot.user_types = ['dns_1k', 'pos', 'fare', 'kosarak']
    # plot.eps('eps_medium', 'query_mse', 65536)
    # plot.eps('eps_medium', 'query_mae', 65536)
    # plot.eps('eps_medium', 'query_mmae', 65536)
    # plot.eps('eps_medium', 'query_mmse', 65536)

    # Smoother comparison
    # plot.user_types = ['dns_1k', 'pos', 'fare', 'kosarak']
    # plot.eps('eps_medium', 'smooth', 65536)

    # Hierarchy comparison
    # plot.user_types = ['dns_1k', 'pos', 'fare', 'kosarak']
    # plot.eps('eps_medium', 'range', 65536)

    # There is no optimal p
    # plot.user_types = ['dns_1k']
    # plot.eps('eps_range_medium', 'ell_est', 65536)
    # plot.eps('eps_range_small', 'ell_est', 65536)
    # plot.eps('eps_range_medium', 'ell_est', 4096)
    # plot.eps('eps_range_small', 'ell_est', 4096)

    # Compare different method to find the threshold
    # plot.user_types = ['dns_1k', 'pos', 'fare', 'kosarak']
    # plot.eps('eps_medium', 'er0_ell_est', 65536)

    # Compare different methods to find threshold in LDP
    # plot.user_types = ['dns_1k', 'pos', 'fare', 'kosarak']
    # plot.eps('eps_ldp', 'er10_ell_est', 65536)

    # ToPL visualization
    # plot.user_types = ['dns_1k']
    # plot.precise('eps_ldp', 65536)
