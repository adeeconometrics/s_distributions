try:
    from scipy.special import erfinv as _erfinv
    from scipy.integrate import quad as _quad
    import numpy as _np
    from math import sqrt as _sqrt, exp as _exp, pi as _pi, log as _log
    from typing import Union, Tuple, List
    from abc import ABC
except Exception as e:
    print(f"some modules are missing {e}")

class Base(ABC):
    def __init__(self):
        if type(self) is Base:
            raise TypeError('Continuous Univariate Base class cannot be instantiated.')

    def logpdf(self, x: Union[List[float], _np.ndarray] = None) -> Union[float, _np.ndarray]:
        """
        Args:

            x (List[float], numpy.ndarray): random variable or list of random variables

        Returns:
            logpdf of Beta prime distribution.
        """

        if x is not None:
            if not (isinstance(x, _np.ndarray)) and issubclass(x, List):
                raise TypeError(f'parameter x only accepts List types or numpy.ndarray')
            else:
                return _np.log(self.pdf(x))
        return _log(self.pdf())

    def logcdf(self, x: Union[List[float], _np.ndarray] = None) -> Union[float, _np.ndarray]:
        """
        Args:

            x (List[float], numpy.ndarray): random variable or list of random variables

        Returns:
            logcdf of Beta prime distribution.
        """

        if x is not None:
            if not (isinstance(x, _np.ndarray)) and issubclass(x, List):
                raise TypeError(f'parameter x only accepts List types or numpy.ndarray')
            else:
                return _np.log(self.cdf(x))
        return _log(self.cdf())

    # test for performance consideration 
    # concrete class should not ask for random variable as ctor parameter
    @classmethod
    def likelihood(cls, theta:Union[List[Tuple], Tuple], 
                        x:Union[List[float], float]) -> Union[float, List[float]]:

        if not (isinstance(theta, (Tuple, List)) and isinstance(x, (List, float))):
            raise TypeError('invalid type parameters')

        if isinstance(theta, Tuple):
            if type(x) is float:
                return cls(*theta).pdf(x)
            return _np.prod(cls(*theta).pdf(x))

        if isinstance(theta, List):
            if type(x) is float:
                return [cls(*_theta).pdf(x) for _theta in theta]
            return [_np.prod(cls(*_theta).pdf(x)) for _theta in theta]

    @classmethod
    def log_likelihood(cls, theta:Union[List[Tuple], Tuple], 
                        x:Union[List[float], float]) -> Union[float, List[float]]:

        if not (isinstance(theta, (Tuple, List)) and isinstance(x, (List, float))):
            raise TypeError('invalid type parameters')
        
        if isinstance(theta, Tuple):
            if type(x) is float:
                return _log(cls(*theta).pdf(x))
            return _np.log(cls(*theta).pdf(x)).sum()

        if isinstance(theta, List):
            if type(x) is float:
                return [_log(cls(*_theta).pdf(x)) for _theta in theta]
            return [_np.log(cls(*_theta).pdf(x)).sum() for _theta in theta]

    def mle(self) -> NotImplemented:
        return NotImplemented

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

    # special functions for ϕ(x), and Φ(x) functions
    @staticmethod
    def stdnorm_pdf(x:float) -> float:
        return _exp(-pow(x, 2)/2) / _sqrt(2*_pi)

    @staticmethod
    def stdnorm_cdf(x:float) -> float:
        return _quad(self.stdnorm_pdf, -_np.inf, x)[0]

    @staticmethod
    def stdnorm_cdf_inv(x:float, p:float, mean:float = 0.0, std:float = 1.0) -> float:
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
