from estimator.pak import PAK
from estimator.ss_hie import SSHie
from estimator.svt_bt import SvtBT
from estimator.svt_hie import SvtHie
from estimator.sw_hm import SWHM


class EstimatorFactory(object):
    @staticmethod
    def create_estimator(name, users, args):
        if name == 'svt_hie':
            estimator = SvtHie(users, args)
        elif name == 'pak':
            estimator = PAK(users, args)
        elif name == 'svt_bt':
            estimator = SvtBT(users, args)
        elif name == 'ss_hie':
            estimator = SSHie(users, args)
        elif name == 'sw_hm':
            estimator = SWHM(users, args)
        else:
            raise NotImplementedError
        return estimator
