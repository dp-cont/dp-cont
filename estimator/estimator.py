import abc


class Estimator(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, users, args):
        self.users = users
        self.args = args

        self.tree = None
        self.tree_range_list = []

        self.num_levels = 0
        self.group_indexs = []
        self.percentile_estimator = None
        self.range_estimator = None

    def estimate(self, p=0):
        ell = self.percentile_estimator.obtain_ell(p)
        est = self.range_estimator.est_precise(ell)
        return est
