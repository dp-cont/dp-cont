from estimator.estimator import Estimator
from percentile_estimator.estimator_factory import PercentileEstimatorFactory
from range_estimator.estimator_factory import RangeEstimatorFactory


class NMBT(Estimator):

    def __init__(self, users, args):
        Estimator.__init__(self, users, args)
        self.percentile_estimator = PercentileEstimatorFactory.create_estimator('nm_mse', users, args)
        self.range_estimator = RangeEstimatorFactory.create_estimator('basic_hierarchy', users, args)
