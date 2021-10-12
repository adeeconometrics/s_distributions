try:
    from scipy.special import erfinv as _erfinv
    from scipy.integrate import quad as _quad
    import matplotlib.pyplot as plt
    from math import sqrt as _sqrt, log as _log, exp as _exp
    from typing import Union
    from abc import ABC
except Exception as e:
    print(f"some modules are missing {e}")

"""
Alternative design routes:
- remove logpdf, logcdf and implement it directly on concrete class
- raise NotImplementedError on functions in this class
- remove plot option and place it elsewhere
- make arguments private and add getter-setter decorators
"""
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
        return _log(pdf)

    def logcdf(self, cdf) -> number:
        return _log(cdf)

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
        return _exp(-pow(x, 2)/2)/_sqrt(2*_pi)

    def stdnorm_cdf(self, x) -> float:
        return _quad(self.stdnorm_pdf, -inf, x)[0]

    def stdnorm_cdf_inv(self, x, p, mean=0, std=1) -> float:
        """
        quantile function of the normal cdf. Note that p can only have values between (0,1).
        `stdnorm_cdf_int` defaults to standard normal but can be expressed more generally.
        """
        return mean + std*_sqrt(2)*_erfinv(2*p-1)
