import abc


class Estimator(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, users, args):
        self.users = users
        self.args = args
        self.epsilon = self.args.epsilon
        self.p = 0

    @abc.abstractmethod
    def obtain_ell(self, p=0):
        pass

    def verbose_plot_quality(self, error_1, error_2, true_mse, plot_folder, plot_name):
        import matplotlib.pyplot as plt
        from matplotlib.pyplot import figure
        plt.rcParams.update({'figure.autolayout': True})
        plt.autoscale()
        plt.figure()
        w, h = 8.7, 4.4
        figure(figsize=(w, h))
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = "Times New Roman"

        plt.rcParams["font.size"] = 20
        plt.yticks(fontsize=20)
        plt.yscale("log")
        plt.plot(- error_1, label='noise', linestyle=':', linewidth=3, color=(0.0392, 0.4627, 0.7333))
        plt.plot(- error_2, linestyle='--', label='bias', linewidth=3, color=(0.9608, 0.7490, 0.2588))
        plt.plot(- error_1 - error_2, label='quality', linewidth=3, color=(0.3529, 0.7373, 0.5451))
        plt.plot(true_mse, label='true', linestyle='-.', linewidth=3)
        plt.legend()
        plt.show()
