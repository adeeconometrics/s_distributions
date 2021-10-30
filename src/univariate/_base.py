try:
    from scipy.special import erfinv as _erfinv
    from scipy.integrate import quad as _quad
    import numpy as _np
    from math import sqrt as _sqrt, exp as _exp, pi as _pi, log as _log
    from typing import Union, Tuple, List
    from abc import ABC, abstractmethod
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
        """
        Generic method for log likelihood.
        
        Args:
            theta (Union[List[Tuple], Tuple]): population parameter 
            x (Union[List[float], float]): data

        Returns:
            log-likelihood of a probability distribution given a data defined in x.
        """

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
        
        """
        Generic method for log likelihood.
        
        Args:
            theta (Union[List[Tuple], Tuple]): population parameter 
            x (Union[List[float], float]): data

        Returns:
            log-likelihood of a probability distribution given a data defined in x.
        """

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

    # def mle(self): # If there exist a generic ML Estimator, keep this.
    #     """
    #     Default implementation of Maximum Likelihood Estimator.
    #     Raise NotImplementedError.
    #     """
    #     raise NotImplementedError('Maximum likelihood Estimator is not implemented.')

    @abstractmethod
    def pdf(self): # guarantee that all concrete class will have a defined pdf
        pass

    @abstractmethod
    def cdf(self): # guarantee that all concrete class will have a defined cdf
        pass

    def pvalue(self):
        """
        Default implementation of p-value.
        Returns NotImplemented.
        """
        return NotImplemented

    def confidence_interval(self): # staged for removing
        """
        Default implementation of confidence interval.
        Returns NotImplemented.
        """
        return NotImplemented

    def rvs(self):  # MH algorithm
        """
        returns random variate samples default NotImplemented
        """
        return "currently unsupported"

    def mean(self):
        """
        Default implementation of the mean.
        Returns NotImplemented.
        """
        return NotImplemented

    def median(self):
        """
        Default implementation of the median.
        Returns NotImplemented.
        """
        return NotImplemented

    def mode(self):
        """
        Default implementation of the mode.
        Returns NotImplemented.
        """
        return NotImplemented

    def var(self):
        """
        Default implementation of the variance.
        Returns NotImplemented.
        """
        return NotImplemented

    def std(self): # make this generic
        """
        Default implementation of the standard deviation.
        Returns NotImplemented.
        """
        return NotImplemented

    def skewness(self):
        """
        Default implementation of skewness.
        Returns NotImplemented.
        """
        return NotImplemented

    def kurtosis(self):
        """
        Default implementation of kurtosis.
        Returns NotImplemented.
        """
        return NotImplemented

    def entropy(self):
        """
        Default implementation of entropy.
        Returns NotImplemented.
        """
        return NotImplemented

    # special functions for ϕ(x), and Φ(x) functions
    @staticmethod
    def stdnorm_pdf(x:float) -> float:
        """
        Generic method for standard normal pdf.

        Args: x(float)
        Return: float
        """
        return _exp(-pow(x, 2)/2) / _sqrt(2*_pi)

    @staticmethod
    def stdnorm_cdf(x:float) -> float:
        """
        Generic method for standard normal cdf.

        Args: x(float)
        Return: float
        """
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
