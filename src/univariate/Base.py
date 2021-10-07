try:
    from typing import Union
    from math import sqrt, pow, log, pi
    from scipy.special import erfinv
    from scipy.integrate import quad
    from abc import ABC
    from numpy import exp, inf
    import matplotlib.pyplot as plt
except Exception as e:
    print(f"some modules are missing {e}")


class Base(ABC):
    def __init__(self, data: Union[list[number], np.ndarray]):
        self.data = data

    # add fill-color function given some condition

    def plot(self, x, y, xlim=None, ylim=None, xlabel=None, ylabel=None):
        if ylim is not None:
            plt.ylim(0, ylim)  # scales from 0 to ylim
        if xlim is not None:
            plt.xlim(-xlim, xlim)
        if xlabel is not None:
            plt.xlabel(xlabel)
        if ylabel is not None:
            plt.ylabel(ylabel)
        plt.plot(x, y, "black", alpha=0.5)
        plt.show()

    def logpdf(self, pdf) -> number:
        return log(pdf)

    def logcdf(self, cdf) -> number:
        return log(cdf)

    def pvalue(self) -> str:
        return "unsupported"

    def confidence_interval(self) -> str:
        return "currently unsupported"

    def rvs(self):  # (adaptive) rejection sampling implementation
        """
        returns random variate samples default (unsupported)
        """
        return "currently unsupported"

    def mean(self) -> str:
        """
        returns mean default (unsupported)
        """
        return "unsupported"

    def median(self) -> str:
        """
        returns median default (unsupported)
        """
        return "unsupported"

    def mode(self) -> str:
        """
        returns mode default (unsupported)
        """
        return "unsupported"

    def var(self) -> str:
        """
        returns variance default (unsupported)
        """
        return "unsupported"

    def std(self) -> str:
        """
        returns the std default (undefined)
        """
        return "unsupported"

    def skewness(self) -> str:
        """
        returns skewness default (unsupported)
        """
        return "unsupported"

    def kurtosis(self) -> str:
        """
        returns kurtosis default (unsupported)
        """
        return "unsupported"

    def entropy(self) -> str:
        """
        returns entropy default (unsupported)
        """
        return "unsupported"

    # special functions for ϕ(x), and Φ(x) functions: should this be reorganized?
    def stdnorm_pdf(self, x) -> float:
        return exp(-pow(x, 2)/2)/sqrt(2*pi)

    def stdnorm_cdf(self, x) -> float:
        return quad(self.stdnorm_pdf, -inf, x)[0]

    def stdnorm_cdf_inv(self, x, p, mean=0, std=1) -> float:
        """
        qunatile function of the normal cdf. Note thatn p can only have values between (0,1).
        defaults to standard normal but can be expressed more generally.
        """
        return mean + std*sqrt(2)*erfinv(2*p-1)
