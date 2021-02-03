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

import config


class GenerateFile:
    def __init__(self):
        pass

    def convert_to_dict(self, data):
        if len(data) == 0:
            return {}

        else:
            layers = len(data[0]) - 1
            dict_data = Vividict()

            for row in data:
                if layers == 1:
                    dict_data[row[0]] = row[1]
                elif layers == 2:
                    dict_data[row[0]][row[1]] = row[2]
                elif layers == 3:
                    dict_data[row[0]][row[1]][row[2]] = row[3]
                elif layers == 4:
                    dict_data[row[0]][row[1]][row[2]][row[3]] = row[4]
                elif layers == 5:
                    dict_data[row[0]][row[1]][row[2]][row[3]][row[4]] = row[5]
                else:
                    raise Exception("number of layers larger than 5")

            return dict_data
        
    def convert_to_dict_std(self, data):
        if len(data) == 0:
            return {}

        else:
            layers = len(data[0]) - 2
            dict_data = Vividict()

            for row in data:
                if layers == 1:
                    dict_data[row[0]] = [row[1], row[2]]
                elif layers == 2:
                    dict_data[row[0]][row[1]] = [row[2], row[3]]
                elif layers == 3:
                    dict_data[row[0]][row[1]][row[2]] = [row[3], row[4]]
                elif layers == 4:
                    dict_data[row[0]][row[1]][row[2]][row[3]] = [row[4], row[5]]
                elif layers == 5:
                    dict_data[row[0]][row[1]][row[2]][row[3]][row[4]] = [row[5], row[6]]
                else:
                    raise Exception("number of layers larger than 5")

            return dict_data

    def generate_json(self, save_data, save_name):
        file_name = config.PLOT_PATH + save_name + ".json"
        write_file = open(file_name, 'w')
        json.dump(save_data, write_file, indent=4)


class Vividict(dict):
    def __missing__(self, key):
        value = self[key] = type(self)()

        return value

    def walk(self):
        for key, value in self.items():
            if isinstance(value, Vividict):
                for tup in value.walk():
                    yield (key,) + tup
            else:
                yield key, value


if __name__ == "__main__":
    pass

