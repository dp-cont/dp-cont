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

import logging
import MySQLdb
import socket

import config


class OperateDatabase:
    def __init__(self):
        self.logger = logging.getLogger("operate_database")
    
    def connect(self, host, user, password, database):
        if socket.gethostname() == "admin-node":
            host = "127.0.0.1"
        
        self.db = MySQLdb.connect(host, user, password, database, charset="utf8")
        
        self.logger.info("database connected")
    
    def close(self):
        self.db.close()
    
    def update(self, table, data):
        cursor = self.db.cursor()

        keys = ', '.join(data.keys())
        values = ', '.join(['%s'] * len(data))

        # sql = 'REPLACE INTO {table}({keys}) VALUES ({values})'.format(table=table, keys=keys, values=values)
        sql = 'INSERT INTO {table}({keys}) VALUES ({values}) ON DUPLICATE KEY UPDATE'.format(table=table, keys=keys, values=values)
        update = ','.join([" {key} = %s".format(key=key) for key in data])
        sql += update

        try:
            cursor.execute(sql, tuple(data.values()) * 2)
            self.db.commit()
        except:
            self.logger.info("Error: updating record failed")
            self.db.rollback()
    
    def query(self, table, select_keys, conditions):
        if type(conditions) is str:
            conditions = [conditions]
        if len(conditions) <= 11:
            conditions = conditions + [True for _ in range(11 - len(conditions))]
        else:
            raise Exception('number of conditions larger than 11')

        cursor = self.db.cursor()

        select_keys = ', '.join(select_keys)
        sql = 'SELECT {select_keys} ' \
              'FROM {table} WHERE {c1} AND {c2} AND {c3} AND {c4} AND {c5} AND {c6} AND {c7} AND {c8} AND {c9} AND {c10} AND {c11}'.format(
               select_keys=select_keys, table=table,
               c1=conditions[0], c2=conditions[1], c3=conditions[2], c4=conditions[3], c5=conditions[4],
               c6=conditions[5], c7=conditions[6], c8=conditions[7], c9=conditions[8], c10=conditions[9],
               c11=conditions[10])
        
        try:
            cursor.execute(sql)
            
            results = cursor.fetchall()
            
            return results
        except:
            self.logger.info("Error: unable to fecth data")


if __name__ == "__main__":
    output_file = None
    logging.basicConfig(filename=output_file,
                        format='%(levelname)s:%(asctime)s: - %(name)s - : %(message)s',
                        level=logging.DEBUG)

    operate_db = OperateDatabase()
    operate_db.connect(config.DATABASE_HOST, config.DATABASE_USER, config.DATABASE_PASSWORD, config.DATABASE_NAME)
    
    table = 'template'
    data = {
        'first_name': 'si',
        'last_name': 'li',
        'age': 25,
        'salary': 8000
    }
    
    operate_db.update(table, data)

    operate_db.close()
