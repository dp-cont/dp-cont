from user.dns_user import DnsUser
from user.taxi_user import TaxiUser
from user.transaction_user import TransactionUser


class UserFactory(object):
    @staticmethod
    def create_user(name, args):
        if name in ['time', 'fare', 'dist']:
            users = TaxiUser(args)
        elif name in ['kosarak', 'pos', 'web1', 'web2', 'aol', 'retail', 'star']:
            users = TransactionUser(args)
        elif name in ['dns_1k', 'google_1k']:
            users = DnsUser(args)
        else:
            raise ValueError('user name %s not defined' % name)
        return users
