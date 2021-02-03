import pickle

from user.user import User


class DnsUser(User):
    # max_ell_dict = {'dns_1k': 617, 'google_1k': 36}
    max_ell_dict = {'dns_1k': 2000, 'google_1k': 500}

    def initial_generate(self):
        user_file_name = 'data/dns/'
        self.data = pickle.load(open(user_file_name + self.args.user_type + '.dat', 'rb'))
        self.data = self.data.astype(int)
        self.n = len(self.data)
        self.max_ell = self.max_ell_dict[self.args.user_type]
