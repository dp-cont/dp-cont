import fcntl
# import MySQLdb
import json
import os.path

import numpy as np
import pickle


# def connect():
#     db = MySQLdb.connect(host='dunes.cs.purdue.edu', port=3306, user='ldp_post_process', passwd='5MlDZmY4fUy1P9Bo',
#                          db='ldp_post_process',
#                          charset='utf8')
#     cursor = db.cursor()
#     return db, cursor
#
#
# def write_data(db, cursor, name, key_list):
#     sql = """
#                       REPLACE INTO comparison SET
#                       epsilon = %s,
#                       sse_mean = %s,
#                       sse_std = %s,
#                       method = %s,
#                       dataset = %s
#                   """
#
#     cursor.execute(sql, (epsilon, error_statistic, error_statistic[0][1], method_name, dataset_name))
#     db.commit()
#     db.close()


def write_file_head(args, filename, meta_info):
    if not args.write_file:
        return

    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            json_data = json.load(f)

            if args.overwrite:
                json_data['meta_info'] = {args.vary: [str(x) for x in meta_info]}
                json_data['results'] = {}
            else:
                # union with the new setting
                existing_meta_info = json_data['meta_info'][args.vary]
                existing_meta_info = sorted(list(set([float(x) for x in existing_meta_info] + list(meta_info))))
                json_data['meta_info'][args.vary] = existing_meta_info
            f.close()

        with open(filename, 'w') as f:
            f.write(json.dumps(json_data, indent=4))
            f.close()

    else:
        json_data = {
            'meta_info': {args.vary: [str(x) for x in meta_info]},
            'results': {}
        }
        with open(filename, 'w') as f:
            f.write(json.dumps(json_data, indent=4))
            f.close()


def unroll(data):
    new_data = {}
    for exp_i, data_round_i in enumerate(data):
        for method, method_data in data_round_i.items():
            if method in new_data:
                new_data[method].append(method_data)
            else:
                new_data[method] = [method_data]
    return new_data


def append_to_file(args, file_names, filename, param):

    if not args.write_file:
        return

    data = []
    for f_i in file_names:
        data.append(pickle.load(open(f_i, 'rb')))
    data = unroll(data)

    with open(filename, 'r+') as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        json_data = json.load(f)
        if args.overwrite:
            json_data['results'][str(param)] = data
        else:
            if str(param) in json_data['results']:
                for method, method_data in data.items():
                    if (method not in json_data['results'][str(param)]) or args.overwrite_method:
                        json_data['results'][str(param)][method] = method_data
                    else:
                        json_data['results'][str(param)][method] += method_data

            else:
                json_data['results'][str(param)] = data
        f.seek(0)
        f.truncate()
        f.write(json.dumps(json_data, indent=4))
        f.close()


def read_value(json_data, varys):
    y_data = []
    y_error = []
    x_data = []

    if json_data:
        for vary in varys:
            y_values = []
            if str(vary) not in json_data:
                continue

            x_data.append(float(vary))
            y_values_temp = json_data[str(vary)]
            for v in y_values_temp:
                y_values += v[1]
            y_data.append(np.mean(y_values))
            y_error.append(np.std(y_values))

    return y_data, y_error, x_data
