try:
    import numpy as _np
    from typing import Union, Tuple, Dict, List
    from math import sqrt as _sqrt, pi as _pi, asin as _asin, log as _log
    from univariate._base import BoundedInterval
except Exception as e:
    print(f"some modules are missing {e}")


class Arcsine(BoundedInterval):
    """
    This class contains methods concerning Arcsine Distirbution [#]_.
    
    .. math::
        \\text{Arcsine}(x)={\\frac{1}{\pi \\sqrt{x(1-x)}}}

    Args:

        x(float): random variable between 0 and 1

    Reference:
    .. [#] Wikipedia contributors. (2020, October 30). Arcsine distribution. In Wikipedia, The Free Encyclopedia.
    Retrieved 05:19, December 30, 2020, from https://en.wikipedia.org/w/index.php?title=Arcsine_distribution&oldid=986131091
    """

    def pdf(self, x: Union[List[float], _np.ndarray, float]) -> Union[float, _np.ndarray]:
        """
        Args:
            x (Union[List[float], _np.ndarray, float]): random variables

        Raises:
            ValueError: when there exist a value less than 0 or greater than 1
            TypeError: when parameter is not of type float | List[float] | numpy.ndarray

        Returns:
            Union[float, _np.ndarray]: evaluation of pdf at x
        """
        if isinstance(x, (_np.ndarray, List)):
            x = _np.array(x)
            if _np.any(x<0 or x>1):
                raise ValueError(f'random variable should have values between [0,1].')
            return 1/(_pi * _np.sqrt(x*(1-x)))

        if type(x) is float:
            if x < 0 or x > 1:
                raise ValueError(f'random variable should have values between [0,1].')
            return 1/_pi*_sqrt(x * (1-x))

        raise TypeError('parameter x is expected to be of type float | List[float] | numpy.ndarray')
        

    def cdf(self, x: Union[List[float], _np.ndarray, float]) -> Union[float, _np.ndarray]:
        """
        Args:
            x (Union[List[float], _np.ndarray, float]): random variables

        Raises:
            ValueError: when there exist a value less than 0 or greater than 1
            TypeError: when parameter is not of type float | List[float] | numpy.ndarray

        Returns:
            Union[float, _np.ndarray]: evaluation of cdf at x
        """
        if isinstance(x, (_np.ndarray, List)):
            x = _np.array(x)
            if _np.any(x<0 or x>1):
                raise ValueError(f'random variable can only be evaluated in the domain [0,1]')
            return 1/(_pi)*_np.arcsin(_np.sqrt(x))

        if type(x) is float:
            if x<0 or x>1:
                raise ValueError(f'random variable can only be evaluated in the domain [0,1]')
            return 1/_pi * _asin(_sqrt(x))

        raise TypeError('parameter x is expected to be of type float | List[float] | numpy.ndarray')


    def mean(self) -> float:
        """
        Returns:
            float: mean of Arcsine distribution.
        """
        return 0.5

    def median(self) -> float:
        """
        Returns:
            float:  median of Arcsine distribution
        """
        return 0.5

    def mode(self) -> Tuple[float, float]:
        """
        Returns:
            Tuple[float, float]: mode of Arcsine distribution
        """
        return (0, 1)

    def var(self) -> float:
        """
        Returns:
            float: variance of Arcsine distribution
        """
        return 0.125

    def std(self) -> float:
        """
        Returns:
            float: standard deviation of Arcsine distribution
        """
        return _sqrt(0.125)

    def skewness(self) -> float:
        """
        Returns:
            float: skewness of Arcsine distribution
        """
        return 0.0

    def kurtosis(self) -> float:
        """
        Returns:
            float: kurtosis of Arcsine distribution
        """
        return 1.5

    def entropy(self)->float:
        """
        Returns:
            float: entropy of Arcsine distribution
        """
        return _log(_pi/4)

    def summary(self) -> Dict[str, Union[float, Tuple[float, float]]]:
        """
        Summary statistic regarding the Arcsine distribution which contains the following parts of the distribution:
        (mean, median, mode, var, std, skewness, kurtosis).

        Returns:
            Dict[str, Union[float, Tuple[float, float]]]
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }
