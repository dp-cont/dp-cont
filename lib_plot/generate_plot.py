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

import os
import logging

from lib_remote.generate_file import GenerateFile
from lib_remote.operate_database import OperateDatabase
import config


class GeneratePlot:
    def __init__(self):
        self.logger = logging.getLogger("generate_plot")

        self.operate_db = OperateDatabase()
        self.operate_db.connect(config.DATABASE_HOST, config.DATABASE_USER, config.DATABASE_PASSWORD, config.DATABASE_NAME)
        self.generate_file = GenerateFile()

    def generate_template(self):
        save_name = 'template'
        query_table = 'template'
        save_data = {}

        for last_name in ['zhang', 'li']:
            select_keys = ['age', 'salary']
            conditions = ["last_name='%s'" % (last_name,)]
            records = self.operate_db.query(query_table, select_keys, conditions)
            records_dict = self.generate_file.convert_to_dict(records)

            save_data[last_name] = records_dict

        self.generate_file.generate_json(save_data, save_name)

    def close_connection(self):
        self.operate_db.close()


def main():
    output_file = None
    logging.basicConfig(filename=output_file,
                        format='%(levelname)s:%(asctime)s: - %(name)s - : %(message)s',
                        level=logging.DEBUG)

    generate_plot = GeneratePlot()

    generate_plot.generate_template()
    
    generate_plot.close_connection()


if __name__ == "__main__":
    os.chdir("../")

    main()
