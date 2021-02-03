import abc


class Estimator(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, users, args):
        self.users = users
        self.args = args

        self.name = None

        self.tree = None
        self.tree_range_list = []

        self.num_levels = 0
        self.group_indexs = []

    @abc.abstractmethod
    def est_precise(self, tau):
        return
