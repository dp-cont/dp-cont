import os
from loguru import logger
import numpy as np
import pickle

from estimator.estimator_factory import EstimatorFactory
from percentile_estimator.estimator_factory import PercentileEstimatorFactory
from range_estimator.estimator_factory import RangeEstimatorFactory


class Evaluator(object):

    def __init__(self, args, users):
        self.args = args
        self.users = users
        self.range_estimator = None

    def eval(self, methods, percentile_methods, range_methods, smooth_methods):
        if self.args.metric == 'ell_est':
            ret = self.eval_ell_est(percentile_methods)
        elif self.args.metric == 'range':
            ret = self.eval_range(range_methods)
        elif self.args.metric == 'smooth':
            ret = self.eval_smooth(smooth_methods)
        elif self.args.metric == 'precise':
            ret = self.eval_precise(methods)
        elif self.args.metric in ['query_mse', 'query_mae', 'query_mmae', 'query_mmse']:
            ret = self.eval_full(methods)
        else:
            raise NotImplementedError(self.args.metric)

        f_name = 'tmp/%s-%d.pkl' % (self.args.metric, os.getpid())
        pickle.dump(ret, open(f_name, 'wb'))
        return f_name

    def eval_ell_est(self, methods):

        n_query = 50

        result_dict = {}
        data_ell = self.users.data[:self.args.m]
        data_hie = data_ell[np.random.choice(self.args.m, 16 * self.args.m, replace=True)]
        self.users.data = np.concatenate((data_ell, data_hie), axis=0)
        self.users.n = len(self.users.data)
        self.users.m = self.args.m

        # large means we are using LDP
        if self.args.vary in ['eps_large', 'eps_ldp']:
            exepected_exp_round = 30
            range_estimator = RangeEstimatorFactory.create_estimator('hm', self.users, self.args)
            range_estimator.epsilon = self.args.range_epsilon
        else:
            exepected_exp_round = 100
            range_estimator = RangeEstimatorFactory.create_estimator('guess_hierarchy', self.users, self.args)
            range_estimator.epsilon = self.args.range_epsilon / range_estimator.num_levels

        self.range_estimator = range_estimator
        zeros = np.zeros_like(data_hie)
        query_indexes = self.my_queries(n_query)

        eval_size = 1 if self.args.exp_round == 1 else int(exepected_exp_round / self.args.exp_round)
        for method in methods:
            percentile_estimator = PercentileEstimatorFactory.create_estimator(method, self.users, self.args)

            mses = np.zeros(eval_size)
            for i in range(eval_size):
                ell = percentile_estimator.obtain_ell(self.args.p)

                if self.args.exp_round == 1:
                    logger.info(f'{method}, {ell}')

                if ell == 0:
                    est = zeros
                else:
                    est = range_estimator.est_precise(ell)
                mses[i] = self.metric(data_hie, est, query_indexes)

            result_dict[method] = mses.tolist()

        # baseline is always outputing 0
        result_dict['baseline'] = [self.metric(data_hie, zeros, query_indexes)]

        return result_dict

    def eval_range(self, methods):
        result_dict = {}

        n_query = 10
        query_indexes = self.my_queries(n_query)

        eval_size = 1 if self.args.exp_round == 1 else int(5 * 30 / self.args.exp_round)

        ell = np.percentile(self.users.data[:self.args.m], 95)
        truth = np.copy(self.users.data[self.args.m:])
        truth[truth > ell] = ell

        for method in methods:
            range_estimator = RangeEstimatorFactory.create_estimator(method, self.users, self.args)
            self.range_estimator = range_estimator

            mses = np.zeros(eval_size)
            for i in range(eval_size):
                est = range_estimator.est_precise(ell)
                mses[i] = self.metric(truth, est, query_indexes)

            result_dict[method] = mses.tolist()

        result_dict['baseline'] = [self.metric(truth, np.zeros_like(truth), query_indexes)]
        return result_dict

    def eval_smooth(self, methods):
        result_dict = {}

        n_query = 50
        query_indexes = self.my_queries(n_query)

        eval_size = 1 if self.args.exp_round == 1 else int(5 * 30 / self.args.exp_round)

        ell = np.percentile(self.users.data[:self.args.m], 95)
        truth = np.copy(self.users.data[self.args.m:])
        truth[truth > ell] = ell

        range_estimator = RangeEstimatorFactory.create_estimator('naive_smoother', self.users, self.args)
        self.range_estimator = range_estimator

        hie_leafs = []
        for i in range(eval_size):
            hie_leafs.append(range_estimator.est_precise(ell))

        for method in methods:
            mses = np.zeros(eval_size)
            for i in range(eval_size):
                est = range_estimator.guess(ell, hie_leafs[i], method)
                mses[i] = self.metric(truth, est, query_indexes)

            result_dict[method] = mses.tolist()

        # result_dict['baseline'] = [self.metric(truth, np.zeros_like(truth), query_indexes)]
        return result_dict

    def eval_full(self, methods):
        result_dict = {}

        n_query = 25
        query_indexes = self.my_queries(n_query)
        truth = self.users.data[self.args.m:]
        eval_size = 1 if self.args.exp_round == 1 else int(5 * 40 / self.args.exp_round)
        eval_size = 1

        for method in methods:
            estimator = EstimatorFactory.create_estimator(method, self.users, self.args)
            self.range_estimator = estimator.range_estimator

            errors = np.zeros(eval_size)
            for i in range(eval_size):
                ests = estimator.estimate(self.args.p)
                errors[i] = self.metric(truth, ests, query_indexes)
            result_dict[method] = errors.tolist()

        result_dict['baseline'] = [self.metric(truth, np.zeros_like(truth), query_indexes)]
        return result_dict

    def eval_precise(self, methods):
        result_dict = {}

        n_query = 120
        interval = (self.args.n - self.args.m) / n_query
        query_indexes = np.zeros((n_query, 2), dtype=np.int)
        for i in range(n_query):
            query_indexes[i] = np.array([int(i * interval), int((i + 1) * interval)])

        for method in methods:
            estimator = EstimatorFactory.create_estimator(method, self.users, self.args)
            self.range_estimator = estimator.range_estimator
            ests = estimator.estimate(self.args.p)
            mses = np.zeros(n_query)
            for i in range(n_query):
                mses[i] = self.my_range_sum(ests, query_indexes[i][0], query_indexes[i][1]) / (query_indexes[i][1] - query_indexes[i][0])
            result_dict[method] = mses.tolist()

        truth = self.users.data[self.args.m:]
        mses = np.zeros(n_query)
        for i in range(n_query):
            mses[i] = self.my_range_sum(truth, query_indexes[i][0], query_indexes[i][1]) / (query_indexes[i][1] - query_indexes[i][0])
        result_dict['truth'] = mses.tolist()

        return result_dict

    ''' auxiliary functions'''
    def metric(self, tru_dist, est_dist, query_indexes=None):
        if self.args.metric in ['query_mse', 'ell_est', 'range', 'smooth']:
            return np.mean(np.array([(sum(tru_dist[query_index[0]:query_index[1]])
                                      - self.my_range_sum(est_dist, query_index[0], query_index[1])) ** 2
                                     for query_index in query_indexes]))
        elif self.args.metric in ['query_mae', 'ell_est_mae', 'range_mae', 'smooth_mae']:
            return np.mean(np.array([np.fabs(sum(tru_dist[query_index[0]:query_index[1]])
                                             - self.my_range_sum(est_dist, query_index[0], query_index[1]))
                                     for query_index in query_indexes]))
        elif self.args.metric in ['query_mmae', 'ell_est_mmae', 'range_mmae', 'smooth_mmae']:
            ret = np.zeros(len(query_indexes))
            for query_index_i, query_index in enumerate(query_indexes):
                true_mean = self.my_range_sum(est_dist, query_index[0], query_index[1]) / (query_index[1] - query_index[0])
                est_mean = np.mean(tru_dist[query_index[0]:query_index[1]])
                ret[query_index_i] = np.fabs(est_mean - true_mean)
            return np.mean(ret)
        elif self.args.metric in ['query_mmse', 'ell_est_mmse', 'range_mmse', 'smooth_mmse']:
            ret = np.zeros(len(query_indexes))
            for query_index_i, query_index in enumerate(query_indexes):
                true_mean = self.my_range_sum(est_dist, query_index[0], query_index[1]) / (query_index[1] - query_index[0])
                est_mean = np.mean(tru_dist[query_index[0]:query_index[1]])
                ret[query_index_i] = (est_mean - true_mean) ** 2
            return np.mean(ret)
        else:
            raise ValueError(self.args.metric)

    def my_ell(self, methods):

        result_dict = {}

        eval_size = 1
        for method in methods:
            estimator = PercentileEstimatorFactory.create_estimator(method, self.users, self.args)
            ells = np.zeros(eval_size)
            for i in range(eval_size):
                ells[i] = estimator.obtain_ell(self.args.p)
                print(method, ells[i])

            result_dict[method] = ells.tolist()

        result_dict['true'] = [np.percentile(self.users.data[:self.args.m], self.args.p * 100)]
        return result_dict

    def my_queries(self, n_query):
        query_indexes = np.zeros((n_query, 2), dtype=np.int)
        for i in range(n_query):
            query_indexes[i] = np.sort(np.random.choice(self.users.n - self.args.m, 2)).astype(np.int)
        return query_indexes

    def my_range_sum(self, h, l, r):
        if np.isscalar(h[0]):
            return sum(h[l:r])
        else:
            return self.hierarchy_range(h, l, r)

    def hierarchy_range(self, h, l, r):
        result = 0
        layer = self.range_estimator.num_levels - 1
        nodes_to_invest = range(self.range_estimator.fanout)

        while nodes_to_invest:
            granularity = self.range_estimator.fanout ** layer
            new_nodes = []
            for node in nodes_to_invest:

                if granularity * (node + 1) <= l or granularity * node >= r:
                    continue

                if granularity * node >= l and granularity * (node + 1) <= r:
                    result += h[layer][node]
                else:
                    new_nodes += range(self.range_estimator.fanout * node, self.range_estimator.fanout * (node + 1))

            nodes_to_invest = new_nodes
            layer -= 1
            if layer < 0:
                break

        return result
