import pickle

from user.user import User


class TaxiUser(User):
    max_ell_dict = {'time': 21600, 'dist': 20000, 'fare': 30000}

    def initial_generate(self):
        user_file_name = 'data/nyctaxi/taxi201801_'
        # time is in second, fare is in cents, dist is in 0.01 mile (but they are all ints)
        self.data = pickle.load(open(user_file_name + self.args.user_type + '.pkl', 'rb'))
        self.data = self.data.astype(int)
        self.n = len(self.data)
        self.max_ell = self.max_ell_dict[self.args.user_type]
