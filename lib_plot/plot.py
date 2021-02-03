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

import json
import os
import logging
from lib_plot.plot_line import PlotLine
import config


class Plot:
    def __init__(self):
        self.logger = logging.getLogger("plot")

    def load_data(self, file_name):
        file_name = config.PLOT_PATH + file_name + ".json"
        self.plot_data = json.load(open(file_name))

    def plot_template(self):
        file_name = 'template'
        
        self.load_data(file_name)

        json_data = self.plot_data
        save_name = "_".join((file_name,))

        plot_error = PlotLine(json_data)
        plot_error.format_error_data()

        plot_error.plot_error(markersize=10, error_bar=False)
        plot_error.format_figure(x_label="x", y_label="y",
                                 x_lim=None, y_lim=None,
                                 plot_legend=True, legend_col=2, figure_title=None, log=False)

        plot_error.save_figure(save_name, local=False)


if __name__ == "__main__":
    os.chdir("../")

    output_file = None
    logging.basicConfig(filename=output_file,
                        format='%(levelname)s:%(asctime)s: - %(name)s - : %(message)s',
                        level=logging.INFO)

    plot = Plot()
    
    plot.plot_template()

