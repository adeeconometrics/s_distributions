try:
    from typing import NewType, Optional
    from abc import ABC
    import numpy as np
    from math import sqrt, pow, log
    import scipy as sp
    import scipy.special as ss
    import matplotlib.pyplot as plt
except Exception as e:
    print("some modules are missing {}".format(e))

T = [int, float, np.int16, np.int32, np.int64,
     np.float16, np.float32, np.float64]
number = NewType('number', T)


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

    def pvalue(self) -> Union[number, str]:
        return "unsupported"

    def confidence_interval(self) -> Union[number, str]:
        return "currently unsupported"

    def rvs(self):  # (adaptive) rejection sampling implementation
        """
        returns random variate samples default (unsupported)
        """
        return "currently unsupported"

    def mean(self) -> Union[number, str]:
        """
        returns mean default (unsupported)
        """
        return "unsupported"

    def median(self) -> Union[number, str]:
        """
        returns median default (unsupported)
        """
        return "unsupported"

    def mode(self) -> Union[number, str]:
        """
        returns mode default (unsupported)
        """
        return "unsupported"

    def var(self) -> Union[number, str]:
        """
        returns variance default (unsupported)
        """
        return "unsupported"

    def std(self) -> Union[number, str]:
        """
        returns the std default (undefined)
        """
        return "unsupported"

    def skewness(self) -> Union[number, str]:
        """
        returns skewness default (unsupported)
        """
        return "unsupported"

    def kurtosis(self) -> Union[number, str]:
        """
        returns kurtosis default (unsupported)
        """
        return "unsupported"

    def entropy(self) -> Union[number, str]:
        """
        returns entropy default (unsupported)
        """
        return "unsupported"

    # special functions for ϕ(x), and Φ(x) functions: should this be reorganized?
    def stdnorm_pdf(self, x) -> number:
        return np.exp(-pow(x, 2)/2)/sqrt(2*np.pi)

    def stdnorm_cdf(self, x) -> number:
        return sp.integrate.quad(self.stdnorm_pdf, -np.inf, x)[0]

    def stdnorm_cdf_inv(self, x, p, mean=0, std=1) -> number:
        """
        qunatile function of the normal cdf. Note thatn p can only have values between (0,1).
        defaults to standard normal but can be expressed more generally.
        """
        return mean + std*sqrt(2)*ss.erfinv(2*p-1)
