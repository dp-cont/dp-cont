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
import numpy as np
import config


class PlotBase:
    def __init__(self):
        pass

    def save_figure(self, save_name, local=True):
        dot_index = save_name.find(".")

        if dot_index != -1:
            save_name = list(save_name)
            save_name[dot_index] = "_"
            save_name = ''.join(save_name)

        plt.tight_layout()

        if not local:
            plt.savefig(config.PLOT_PATH + "figures/" + save_name + ".pdf")
        else:
            plt.savefig(config.OVERLEAF_PATH + save_name + ".pdf")


if __name__ == "__main__":
    pass
