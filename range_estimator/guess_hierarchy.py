from range_estimator.smooth_hierarchy import SmoothHierarchy


class GuessHierarchy(SmoothHierarchy):

    def est_precise(self, ell):
        count = self.est_hierarchy(ell)
        count = self.consist(count)
        # naive is "recent" in the paper
        return self.guess(ell, count[0], 'naive_smoother')
