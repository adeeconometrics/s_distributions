try:
    import matplotlib.pyplot as plt
except Exception as e:
    print("some modules are missing {}".format(e))


class Base:  # add histograms
    def __init__(self, data):
        self.data = data

    def scatter(self, x, y, xlim=None, ylim=None, xlabel=None, ylabel=None):
        if ylim is not None:
            plt.ylim(0, ylim)  # scales from 0 to ylim
        if xlim is not None:
            plt.xlim(-xlim, xlim)
        if xlabel is not None:
            plt.xlabel(xlabel)
        if ylabel is not None:
            plt.ylabel(ylabel)
        plt.scatter(x, y)

    # def hist(self, x, y, xlim=None, ylim=None, xlabel=None, ylabel=None):
    #     if ylim is not None:
    #         plt.ylim(0, ylim)  # scales from 0 to ylim
    #     if xlim is not None:
    #         plt.xlim(-xlim, xlim)
    #     if xlabel is not None:
    #         plt.xlabel(xlabel)
    #     if ylabel is not None:
    #         plt.ylabel(ylabel)
    #     plt.scatter(x, y)
