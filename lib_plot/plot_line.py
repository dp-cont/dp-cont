"""
Copyright (C) 2020  Zhikun, Zhang <zhangzhk1993@gmail.com>
Author: Zhikun, Zhang <zhangzhk1993@gmail.com>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from collections import defaultdict

import matplotlib.pyplot as plt

from lib_plot.plot_base import PlotBase


class PlotLine(PlotBase):
    def __init__(self, json_data=None):
        super(PlotLine, self).__init__()

        self.json_data = json_data

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
            "<",
            "+"
        ]

        self.color_list = [
            "#5372ab",
            "#6aa56e",
            "#b75555",
            "#7e74ae",
            "#c9b97d",
            "#000000",
            "purple",
            "orange",
            "royalblue",
            "brown",
            "pink"
        ]

    def format_error_data(self, legend_list=None, x_list=None, std=False):
        self.legends = defaultdict(list)

        if legend_list is None:
            legend_list = self.json_data.keys()

        for legend in legend_list:
            x, y, e = [], [], []

            if legend in self.json_data:
                data = self.json_data[legend]
                data = {float(key): value for key, value in data.items()}

                for x_label in sorted(data):
                    if x_list is None or x_label in x_list:
                        error = data[x_label]

                        x.append(x_label)

                        if std:
                            y.append(error[0])
                            e.append(error[1])
                        else:
                            y.append(error)
                            e.append(0.0)

                self.legends["x"].append(x)
                self.legends["y"].append(y)
                self.legends["e"].append(e)
                self.legends["legend"].append(legend)
            else:
                self.legends["x"].append(x)
                self.legends["y"].append(y)
                self.legends["e"].append(e)
                self.legends["legend"].append(None)

    def plot_error(self, markersize=20, error_bar=True, legend_list=None):
        plt.close()
        plt.figure()
        plt.style.use('seaborn-darkgrid')
        # plt.rc('text', usetex=True)

        if legend_list is None:
            legend_list = self.legends["legend"]

        for index, legend in enumerate(legend_list):
            if legend_list:
                if error_bar:
                    (_, caps, _) = plt.errorbar(self.legends["x"][index], self.legends["y"][index], self.legends["e"][index],
                                                label=legend, color=self.color_list[index], linewidth=2.0,
                                                marker=self.marker_list[index], markersize=markersize, fillstyle='none',
                                                markeredgewidth=2.0)

                    for cap in caps:
                        cap.set_markeredgewidth(2)
                else:
                    plt.plot(self.legends["x"][index], self.legends["y"][index],
                             label=legend, color=self.color_list[index], linewidth=2.0,
                             marker=self.marker_list[index], markersize=markersize, fillstyle='none',
                             markeredgewidth=2.0)

    def plot_legend(self, legend_list, save_name, markersize=10, legend_col=5, local=True):
        plt.close()
        plt.figure()
        plt.style.use('seaborn-darkgrid')
        # plt.rc('text', usetex=True)

        fig = plt.gcf()
        fig.set_size_inches(13.0, 0.8)

        for index, legend in enumerate(legend_list):
            x_data = [2, 2, 2]
            y_data = [2, 2, 2]

            plt.plot(x_data, y_data, label=legend_list[index],
                     color=self.color_list[index], linewidth=2.0,
                     marker=self.marker_list[index], markersize=markersize,
                     fillstyle='none', markeredgewidth=2.0)

        plt.legend(fontsize=15, ncol=legend_col, loc="center")
        plt.ylim([0.0, 1.0])
        plt.axis("off")
        self.save_figure(save_name, local=local)

    def format_figure(self, x_label, y_label, x_lim=None, y_lim=None,
                      plot_legend=True, legend_col=2, legend_loc="best",
                      figure_title=None, log=True):

        if figure_title: plt.title(figure_title)

        if plot_legend: plt.legend(fontsize=15, ncol=legend_col, loc=legend_loc)

        plt.xlim(x_lim)
        plt.ylim(y_lim)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)

        if log: plt.yscale("log")

        plt.xlabel(x_label, fontsize=25)
        plt.ylabel(y_label, fontsize=20)

        # plt.show()


if __name__ == "__main__":
    pass
