from estimator.pak import PAK
from estimator.ss_hie import SSHie
from estimator.nm_bt import NMBT
from estimator.nm_hie import NMHie
from estimator.sw_hm import SWHM


class EstimatorFactory(object):
    @staticmethod
    def create_estimator(name, users, args):
        if name == 'svt_hie':
            estimator = NMHie(users, args)
        elif name == 'pak':
            estimator = PAK(users, args)
        elif name == 'svt_bt':
            # originally named it 'svt' but should be noisy-max
            estimator = NMBT(users, args)
        elif name == 'ss_hie':
            estimator = SSHie(users, args)
        elif name == 'sw_hm':
            estimator = SWHM(users, args)
        else:
            raise NotImplementedError
        return estimator
