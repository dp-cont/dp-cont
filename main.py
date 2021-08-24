import argparse
from loguru import logger
import multiprocessing as mp
import sys
import numpy as np

import recorder
from evaluator import Evaluator
from user.user_factory import UserFactory

q = mp.Queue()

##################################
# uncomment the following to run #
##################################
# ToPS visualization
# general_main('eps_medium', 'precise', user_types=['dns_1k'], exp_round=1, methods=['svt_hie'])
# general_main('eps', 'precise', user_types=['dns_1k'], exp_round=1, methods=['pak'])

# Overall performance comparison
# general_main('eps_medium', 'query_mse', methods=['svt_hie', 'ss_hie', 'svt_bt', 'pak'])

# Smoother comparison
# general_main('eps_medium', 'smooth',)

# Hierarchy comparison
# general_main('eps_medium', 'range',)

# There is no optimal p
# general_main('eps_range_medium', 'ell_est', user_types=['dns_1k'], percentile_methods=['np_p85', 'np_p90', 'np_p95', 'np_p99.5', 'np_p99.9', 'em_mse'])
# general_main('eps_range_small', 'ell_est', user_types=['dns_1k'], percentile_methods=['np_p85', 'np_p90', 'np_p95', 'np_p99.5', 'np_p99.9', 'em_mse'])
# general_main('eps_range_medium', 'ell_est', m=4096, user_types=['dns_1k'], percentile_methods=['np_p85', 'np_p90', 'np_p95', 'np_p99.5', 'np_p99.9', 'em_mse'])
# general_main('eps_range_small', 'ell_est', m=4096, user_types=['dns_1k'], percentile_methods=['np_p85', 'np_p90', 'np_p95', 'np_p99.5', 'np_p99.9', 'em_mse'])

# Compare different method to find the threshold
# general_main('eps_medium', 'ell_est', percentile_methods=['nm_mse', 'smooth', 'smooth_pak'])

# Compare different method to find threshold in LDP
# general_main('eps_ldp', 'ell_est', range_epsilon=1, percentile_methods=['sw_mse', 'sws_mse'])

# ToPL visualization
# general_main('eps_ldp', 'precise', user_types=['dns_1k'], exp_round=1, methods=['sw_hm'])


def general_main(vary, metric,
                 user_types=('dns_1k', 'pos', 'fare', 'kosarak'),
                 epsilon=0.05, range_epsilon=0.05, m=65536, exp_round=10,
                 methods=None, percentile_methods=None, range_methods=None, smooth_methods=None
                 ):
    args.metric = metric
    args.exp_round = exp_round
    args.epsilon = epsilon
    args.range_epsilon = range_epsilon
    for config_i, user_type in enumerate(user_types):
        args.user_type = user_type
        args.m = m
        args.n = 2 ** 20 + args.m

        args.vary = vary
        logger.info('=== vary %s on %s (%d / %d) ===' % (args.vary, args.user_type, config_i + 1, len(user_types)))
        main(methods, percentile_methods, range_methods, smooth_methods)


def main(methods, percentile_methods, range_methods, smooth_methods):
    for arg in vars(args): print(arg, '=', getattr(args, arg), end=', ')
    print('# %s' % (' '.join(sys.argv),))

    if args.vary in ['eps', 'eps_range']:
        user_str = '%s_m%d' % (args.user_type, args.m)
        varys = np.round(np.linspace(0.1, 1, 10), 2)
    elif args.vary in ['eps_ldp', 'eps_range_ldp']:
        user_str = '%s_m%d' % (args.user_type, args.m)
        varys = np.round(np.linspace(0.2, 2, 10), 1)
    elif args.vary in ['eps_medium', 'eps_range_medium']:
        user_str = '%s_m%d' % (args.user_type, args.m)
        varys = np.round(np.linspace(0.01, 0.1, 10), 3)
    else:
        raise NotImplementedError(args.vary)

    if args.metric == 'ell_est' and 'range' not in args.vary:
        user_str += '_er%d' % (10 * args.range_epsilon)

    filename = 'results/%s/%s_%s.json' % (args.vary, user_str, args.metric)
    recorder.write_file_head(args, filename, varys)
    users = UserFactory.create_user(args.user_type, args)

    for i, param in enumerate(varys):
        if args.vary in ['eps', 'eps_medium', 'eps_small', 'eps_large', 'eps_ldp']:
            args.epsilon = param
        elif args.vary in ['eps_range', 'eps_range_medium', 'eps_range_small', 'eps_range_large', 'eps_range_ldp']:
            args.range_epsilon = param
        elif args.vary in ['p']:
            args.p = param

        logger.info('%s = %s (%d / %d)' % (args.vary, param, i + 1, len(varys)))
        parallel_run(users, filename, param, methods, percentile_methods, range_methods, smooth_methods)


def parallel_run(users, filename, param, methods, percentile_methods, range_methods, smooth_methods):
    evaluator = Evaluator(args, users)

    def local_process(evaluator, q):
        np.random.seed()
        result = evaluator.eval(methods, percentile_methods, range_methods, smooth_methods)
        q.put(result)

    q = mp.Queue()
    processes = [mp.Process(target=local_process, args=(evaluator, q)) for _ in range(args.exp_round)]
    for p in processes: p.start()
    for p in processes: p.join()

    results = [q.get() for _ in processes]

    recorder.append_to_file(args, results, filename, param)


parser = argparse.ArgumentParser(description='Exp of ToPS/ToPL')

# method parameter
parser.add_argument('--epsilon', type=float, default=0.05,
                    help='specify the differential privacy parameter, epsilon')
parser.add_argument('--range_epsilon', type=float, default=1,
                    help='specify epsilon for hierarchy (if not the same as the overall epsilon)')
parser.add_argument('--p', type=float, default=0.995,
                    help='specify the percentile')
parser.add_argument('--s', type=float, default=1,
                    help='specify the step size of NM/EM/SVT (default 1)')
parser.add_argument('--m', type=int, default=65536,
                    help='specify the number of users to hold')
parser.add_argument('--g', type=int, default=0,
                    help='specify the guessed leaf nodes (if 0, calculate online)')
parser.add_argument('--r', type=int, default=1048576,
                    help='specify the max interested range')
parser.add_argument('--hie_fanout', type=int, default=16,
                    help='specify the fanout of the hierarchy')
parser.add_argument('--exp_smooth_a', type=float, default=0.6,
                    help='specify the parameter of exponential smoothing')

# experiment parameter
parser.add_argument('--vary', type=str, default='none',
                    help='specify which parameter to vary')
parser.add_argument('--exp_round', type=int, default=16,
                    help='specify the number of iterations for varying frequency')
parser.add_argument('--write_file', type=bool, default=True,
                    help='write to file or not?')
parser.add_argument('--overwrite', type=bool, default=False,
                    help='overwrite existing results or append to existing results?')
parser.add_argument('--overwrite_method', type=bool, default=True,
                    help='overwrite existing results of a specific method (or append)')
parser.add_argument("--multi_process", type=bool, default=True,
                    help="whether to run single-process or multiple")
parser.add_argument("--metric", type=str, default='mse',
                    help="evaluation metric")
parser.add_argument('--user_type', type=str, default='adult',
                    help='specify the type of the data [synthesize, password, url]')

args = parser.parse_args()
