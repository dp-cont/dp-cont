from range_estimator.hierarchy import Hierarchy


class ConsistHierarchy(Hierarchy):

    def __init__(self, users, args):
        Hierarchy.__init__(self, users, args)
        self.fanout = 16
        self.update_fanout()

    def est_precise(self, ell):
        count = self.est_hierarchy(ell)
        count = self.consist(count)
        return count[0]

