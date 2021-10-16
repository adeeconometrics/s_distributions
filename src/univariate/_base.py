try:
    from numpy import inf as _inf, ndarray as _ndarray
    from scipy.special import erfinv as _erfinv
    from scipy.integrate import quad as _quad
    import matplotlib.pyplot as plt
    from math import sqrt as _sqrt, log as _log, exp as _exp, pi as _pi
    from typing import Union
    from abc import ABC
except Exception as e:
    print(f"some modules are missing {e}")

"""
Alternative design routes:
- remove logpdf, logcdf and implement it directly on concrete class
- remove plot option and place it elsewhere
- make arguments private and add getter-setter decorators
"""


class Base(ABC):
    def __init__(self, data: Union[list[Union[int, float]], _ndarray]):
        if type(self) is Base:
            raise TypeError(
                'Continuous Univariate Base class cannot be instantiated.')

        self.data = data

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

    def logpdf(self, pdf) -> Union[int, float]:
        return _log(pdf)

    def logcdf(self, cdf) -> Union[int, float]:
        return _log(cdf)

    def pvalue(self) -> NotImplemented:
        return NotImplemented

    def confidence_interval(self) -> NotImplemented:
        return NotImplemented

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

    # special functions for ϕ(x), and Φ(x) functions: should this be reorganized?
    def stdnorm_pdf(self, x) -> float:
        return _exp(-pow(x, 2)/2) / _sqrt(2*_pi)

    def stdnorm_cdf(self, x) -> float:
        return _quad(self.stdnorm_pdf, -_inf, x)[0]

    def stdnorm_cdf_inv(self, x, p, mean=0, std=1) -> float:
        """
        quantile function of the normal cdf. Note that p can only have values between (0,1).
        `stdnorm_cdf_int` defaults to standard normal but can be expressed more generally.
        """
        return mean + std*_sqrt(2)*_erfinv(2*p-1)


class Infinite(Base):
    """
    Description:
        Base class for probability tags.
    """

    def __init__(self):
        if type(self) is Infinite:
            raise TypeError('base class cannot be instantiated.')


class SemiInfinite(Base):
    """
    Description:
        Base class for probability tags.
    """

    def __init__(self):
        if type(self) is SemiInfinite:
            raise TypeError('base class cannot be instantiated.')


class BoundedInterval(Base):
    """
    Description:
        Base class for probability tags.
    """

    def __init__(self):
        if type(self) is BoundedInterval:
            raise TypeError('base class cannot be instantiated.')
