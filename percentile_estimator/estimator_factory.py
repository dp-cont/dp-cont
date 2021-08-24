from percentile_estimator.em_mse import EMMSE
from percentile_estimator.nm_mse import NMMSE
from percentile_estimator.pf_mse import PFMSE
from percentile_estimator.laplace import Laplace
from percentile_estimator.np_p import NPP
from percentile_estimator.smooth import Smooth
from percentile_estimator.smooth_pak import SmoothPAK
from percentile_estimator.sw import SW
from percentile_estimator.sw_mse import SWMSE
from percentile_estimator.sws import SWS
from percentile_estimator.sws_mse import SWSMSE


class PercentileEstimatorFactory(object):
    @staticmethod
    def create_estimator(name, users, args):
        if name == 'laplace':
            estimator = Laplace(users, args)
        elif name == 'smooth':
            estimator = Smooth(users, args)
        elif name == 'smooth_pak':
            estimator = SmoothPAK(users, args)
        elif name == 'em_mse':
            estimator = EMMSE(users, args)
        elif name == 'nm_mse':
            estimator = NMMSE(users, args)
        elif name == 'pf_mse':
            estimator = PFMSE(users, args)
        elif name[:4] == 'np_p':
            estimator = NPP(users, args)
            if len(name) > 4:
                p = float(name[4:])
                estimator.p = p
        elif name == 'sw':
            estimator = SW(users, args)
        elif name == 'sws':
            estimator = SWS(users, args)
        elif name == 'sw_mse':
            estimator = SWMSE(users, args)
        elif name == 'sws_mse':
            estimator = SWSMSE(users, args)
        else:
            raise NotImplementedError(name)
        return estimator
