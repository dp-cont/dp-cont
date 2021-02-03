from range_estimator.smooth_hierarchy import SmoothHierarchy
from range_estimator.guess_hierarchy import GuessHierarchy
from range_estimator.opt_fanout_hierarchy import OptFanoutHierarchy
from range_estimator.consist_hierarchy import ConsistHierarchy
from range_estimator.basic_hierarchy import BasicHierarchy
from range_estimator.sr import SR
from range_estimator.pm import PM
from range_estimator.hm import HM


class RangeEstimatorFactory(object):
    @staticmethod
    def create_estimator(name, users, args):
        if name == 'opt_fanout_hierarchy':
            estimator = OptFanoutHierarchy(users, args)
        elif name == 'consist_hierarchy':
            estimator = ConsistHierarchy(users, args)
        elif name == 'guess_hierarchy':
            estimator = GuessHierarchy(users, args)
        elif name in ['naive_smoother', 'mean_smoother', 'median_smoother', 'moving_smoother', 'exp_smoother']:
            estimator = SmoothHierarchy(users, args)
        elif name == 'basic_hierarchy':
            estimator = BasicHierarchy(users, args)
        elif name == 'sr':
            estimator = SR(users, args)
        elif name == 'pm':
            estimator = PM(users, args)
        elif name == 'hm':
            estimator = HM(users, args)
        else:
            raise NotImplementedError(name)
        estimator.name = name
        return estimator
