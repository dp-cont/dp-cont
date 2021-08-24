import abc

import numpy as np


class User(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, args):
        self.args = args

        self.data = np.array([])
        self.up_groups = np.array([])
        self.up_groups_dist = []
        self.m_pdf = []
        self.m = self.args.m
        self.n = 0
        self.num_up_group = 0
        self.max_ell = 0

        self.initial_generate()
        self.synopsis_generate()

    def synopsis_generate(self):

        while self.n < self.args.n:
            self.data = np.append(self.data, self.data[self.args.n - self.n:0:-1])
            self.n = len(self.data)
        if self.n > self.args.n:
            self.data = self.data[:self.args.n]
        self.n = self.args.n

        self.m_pdf, _ = np.histogram(self.data[:self.m], bins=range(self.max_ell + 1))

        if self.args.metric == 'up_ell':
            self.num_up_group = int((self.n - self.m) / self.args.n_up_group)
            self.num_up_group = 12
            # omit the first m and final dividant
            self.up_groups = np.split(self.data[self.m: self.m + self.num_up_group * self.args.n_up_group],
                                      self.num_up_group)
            for group_i in range(self.num_up_group):
                n_pdf, _ = np.histogram(self.up_groups[group_i], bins=range(self.max_ell))
                self.up_groups_dist.append(n_pdf)

    @abc.abstractmethod
    def initial_generate(self):
        return
