from range_estimator.hierarchy import Hierarchy


class OptFanoutHierarchy(Hierarchy):

    def __init__(self, users, args):
        Hierarchy.__init__(self, users, args)
        self.fanout = 16
        self.update_fanout()

    def est_precise(self, ell):
        count = self.est_hierarchy(ell)
        return count
