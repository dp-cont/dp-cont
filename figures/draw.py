import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import figure


class Plot:
    def __init__(self):

        self.hatches = ['\\', ' ', 'X', 'o']
        self.opacity = 0.85
        fontP = FontProperties()
        # fontP.set_size('')

        # rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica'], 'size': 16})
        # rc('text', usetex=True)

        self.folder = './'
        self.marker_list = [
            "^",
            "o",
            "s",
            "D",
            "x",
            "*",
            "p",
            "v",
            ">",
            "+"
        ]
        self.color_list = [
            "#5372ab",
            "#6aa56e",
            "#b75555",
            "#7e74ae",
            "purple",
            "red",
            "#c9b97d",
            "#8B4513",
            "#6B8E23",
            "#000000",
        ]

        plt.figure()
        plt.style.use('seaborn-darkgrid')

        self.default_w, self.default_h = 6.4, 4.8
        self.w, self.h = 6.4, 2.6
        figure(figsize=(self.w, self.h))
        self.size_mul = min(self.h / self.default_h, self.w / self.default_w)
        self.size_mul = 1

        font_size = 14
        plt.rcParams["font.size"] = font_size
        self.xlabel_font_size = font_size * self.size_mul
        self.legend_size = font_size * self.size_mul
        self.yticks_size = font_size * self.size_mul
        # print(plt.rcParams.keys())

    # auxiliary functions
    @staticmethod
    def load_data(name, k):
        if not os.path.isfile('../results/%s.json' % name):
            print(name + ' does not exist')
            return None, [None]

        with open('../results/%s.json' % name) as f:
            print(name)
            json_data = json.load(f)

        value_list = list(json_data['meta_info'][k])

        return json_data['results'], value_list

    @staticmethod
    def read_value(json_data, pars, method, div):

        x_data = [float(x) for x in pars]
        y_data = []
        y_error = []

        if json_data:

            for par in pars:
                y_values_temp = json_data[str(par)][method]
                y_values = np.zeros(len(y_values_temp))
                for i in range(len(y_values_temp)):
                    y_values[i] = np.mean(y_values_temp[i]) / div

                y_data.append(np.nanmean(y_values))
                y_error.append(np.nanstd(y_values))

        return x_data, y_data, y_error

    def draw_fig(self, name, fig_name):
        plt.yticks(fontsize=self.yticks_size)
        self.save_fig(name, fig_name)

    def save_fig(self, name, fig_name):
        folder = self.folder + name
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(folder + fig_name + ".pdf", bbox_inches='tight')
        print('save to ' + folder + fig_name)
        # plt.show()
        plt.cla()

    def plot_errorbar(self, x_data, y_data, y_error, method_name, marker_index):
        # print(method_name)
        # print(x_data)
        # print(y_data)
        plt.errorbar(x_data, y_data, [[0] * len(y_error), y_error], label=method_name,
                     color=self.color_list[marker_index], linewidth=2.0 * self.size_mul,
                     marker=self.marker_list[marker_index], markersize=10 * self.size_mul,
                     fillstyle='none', markeredgewidth=2.0 * self.size_mul,
                     )

    def plot_line(self, x_data, y_data, method_name, marker_index):
        # print(method_name)
        # print(x_data)
        # print(y_data)
        plt.plot(x_data, y_data, label=method_name,
                 color=self.color_list[marker_index], linewidth=2.0 * self.size_mul,
                 marker=self.marker_list[marker_index], markersize=1,
                 fillstyle='none', markeredgewidth=2.0 * self.size_mul,
                 )
