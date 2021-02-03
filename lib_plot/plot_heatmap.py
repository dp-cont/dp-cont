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


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from lib_plot.plot_base import PlotBase


class PlotHeatmap(PlotBase):
    def __init__(self, json_data):
        super(PlotHeatmap, self).__init__()

        self.json_data = json_data

    def format_data(self, row_list=None, row_name=None, col_list=None, col_name=None):
        if row_list is None and col_list is None:
            self.df = pd.DataFrame(self.json_data)
        else:
            if row_name is None:
                row_name = row_list
            if col_name is None:
                col_name = col_list

            self.df = pd.DataFrame(data=np.zeros((len(row_name), len(col_name))), columns=col_name, index=row_name)

            for col, col_n in zip(col_list, col_name):
                for row, row_n in zip(row_list, row_name):
                    try:
                        self.df.loc[row_n, col_n] = self.json_data[col][row]
                    except:
                        pass

    def plot_heatmap(self):
        plt.close()
        plt.figure()

        cmap = sns.cubehelix_palette(start=0, rot=1, gamma=0.8, as_cmap=True)
        sns.heatmap(self.df, cmap=cmap, annot=True, fmt=".4f")

    def format_figure(self, x_label, y_label):
        # plt.title(dataset)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlabel(x_label, fontsize=20)
        plt.ylabel(y_label, fontsize=20)


if __name__ == "__main__":
    pass
