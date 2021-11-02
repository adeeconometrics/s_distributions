try:
    import numpy as _np
    from typing import Union, Tuple, Dict, List
    from math import sqrt as _sqrt, pi as _pi, asin as _asin
    from univariate._base import BoundedInterval
except Exception as e:
    print(f"some modules are missing {e}")


class Arcsine(BoundedInterval):
    """
    This class contains methods concerning Arcsine Distirbution.
    Args:

        randvar(float in [0, 1]): random variable

    Reference:
    - Wikipedia contributors. (2020, October 30). Arcsine distribution. In Wikipedia, The Free Encyclopedia.
    Retrieved 05:19, December 30, 2020, from https://en.wikipedia.org/w/index.php?title=Arcsine_distribution&oldid=986131091
    """

    def __init__(self, randvar: Union[float, int]):
        if randvar < 0 or randvar > 1:
            raise ValueError(
                f'random variable should have values between [0,1]. The value of randvar was: {randvar}')

        self.randvar = randvar

    def pdf(self, x: Union[List[float], _np.ndarray, float] = None) -> Union[float, _np.ndarray]:
        """
        Args:

            x (List[float], numpy.ndarray): random variable or list of random variables

        Returns:
            either probability density evaluation for some point or plot of Arcsine distribution.
        """

        if x is not None:
            if not isinstance(x, (_np.ndarray, List)):
                raise TypeError('parameter x only accepts List types or numpy.ndarray')
            else:
                x = _np.array(x)
                return 1/(_pi * _np.sqrt(x*(1-x)))

        return 1/_pi*_sqrt(self.randvar * (1-self.randvar))

    def cdf(self, x: Union[List[float], _np.ndarray] = None) -> Union[float, _np.ndarray]:
        """
        Args:

            x (List[float], numpy.ndarray): random variable or list of random variables

        Returns:
            either cumulative distribution evaluation for some point or plot of Arcsine distribution.
        """

        if x is not None:
            if not isinstance(x, (_np.ndarray, List)):
                raise TypeError('parameter x only accepts List types or numpy.ndarray')
            else:
                x = _np.array(x)
                return 1/(_pi)*_np.arcsin(_np.sqrt(x))

        return 1/_pi * _asin(_sqrt(self.randvar))

    def mean(self) -> float:
        """
        Returns: Mean of the Arcsine distribution.
        """
        return 0.5

    def median(self) -> float:
        """
        Returns: Median of the Arcsine distribution.
        """
        return 0.5

    def mode(self) -> Tuple[float, float]:
        """
        Returns: Mode of the Arcsine distribution. Mode is within the set {0,1}
        """
        return (0, 1)

    def var(self) -> float:
        """
        Returns: Variance of the Arcsine distribution.
        """
        return 0.125

    def std(self) -> float:
        """
        Returns: Standard deviation of the Arcsine distribution.
        """
        return _sqrt(0.125)

    def skewness(self) -> float:
        """
        Returns: Skewness of the Arcsine distribution.
        """
        return 0.0

    def kurtosis(self) -> float:
        """
        Returns: Kurtosis of the Arcsine distribution.
        """
        return 1.5

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
