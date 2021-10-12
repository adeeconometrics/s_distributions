try:
    import matplotlib.pyplot as plt
except Exception as e:
    print(f"some modules are missing {e}")


class Base:  # add histograms
    def __init__(self, data):
        if type(self) is Base:
            raise TypeError('Discrete Univariate Base class cannot be instantiated.')

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

    def pvalue(self) -> NotImplemented:
        return NotImplemented

    def confidence_interval(self) -> NotImplemented:
        return "currently unsupported"

    def rvs(self):  # (adaptive) rejection sampling implementation
        """
        returns random variate samples default NotImplemented
        """
        return "currently unsupported"

    def mean(self) -> NotImplemented:
        """
        returns mean default NotImplemented
        """
        return NotImplemented

    def median(self) -> NotImplemented:
        """
        returns median default NotImplemented
        """
        return NotImplemented

    def mode(self) -> NotImplemented:
        """
        returns mode default NotImplemented
        """
        return NotImplemented

    def var(self) -> NotImplemented:
        """
        returns variance default NotImplemented
        """
        return NotImplemented

    def std(self) -> NotImplemented:
        """
        returns the std default (undefined)
        """
        return NotImplemented

    def skewness(self) -> NotImplemented:
        """
        returns skewness default NotImplemented
        """
        return NotImplemented

    def kurtosis(self) -> NotImplemented:
        """
        returns kurtosis default NotImplemented
        """
        return NotImplemented

    def entropy(self) -> NotImplemented:
        """
        returns entropy default NotImplemented
        """
        return NotImplemented

class Finite(Base):
    """
    Description:
        Base class for probability tags.
    """
    def __init__(self):
        if type(self) is Infinite:
            raise TypeError('base class cannot be instantiated.')

class Infinite(Base):
    """
    Description:
        Base class for probability tags.
    """
    def __init__(self):
        if type(self) is Infinite:
            raise TypeError('base class cannot be instantiated.')