import pickle
from os import path

import numpy as np

from user.user import User


class TransactionUser(User):
    max_ell_dict = {'kosarak': 42178, 'pos': 1638, 'aol': 2290685, 'retail': 16470, 'web1': 497, 'web2': 3340,
                    'star': 2088}

    def initial_generate(self):
        user_file_name = 'data/transaction/%s' % self.args.user_type
        if not path.exists(user_file_name + '_length' + '.pkl'):
            self.clean(user_file_name)

        self.max_ell = self.max_ell_dict[self.args.user_type]
        self.data = pickle.load(open(user_file_name + '_length' + '.pkl', 'rb'))
        self.n = len(self.data)

    def clean(self, user_file_name):
        f = open('%s.dat' % user_file_name, 'r')
        lengths = []
        for line in f:
            if len(line) == 0:
                break
            if line[0] == '#':
                continue
            queries = line.split(' ')
            lengths.append(len(queries))
        lengths = np.array(lengths)
        np.random.shuffle(lengths)
        pickle.dump(lengths, open(user_file_name + '_length' + '.pkl', 'wb'))
