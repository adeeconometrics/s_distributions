try:
    import numpy as _np
    from scipy.special import gammainc as _gammainc
    from math import sqrt as _sqrt, ceil as _ceil, floor as _floor, log2 as _log2
    from typing import Union, Tuple, Dict, List
    from discrete._base import Infinite
except Exception as e:
    print(f"some modules are missing {e}")


class Poisson(Infinite):
    """
    This class contains methods for evaluating some properties of the poisson distribution. 
    As lambda increases to sufficiently large values, the normal distribution (λ, λ) may be used to 
    approximate the Poisson distribution [#]_ [#]_ [#]_.

    Use the Poisson distribution to describe the number of times an event occurs in a finite observation space.

    .. math:: \\text{Poisson}(x;\lambda) = \\frac{\lambda ^{x} e^{- \lambda}}{x!}

    Args: 
        λ (float): expected rate if occurrences.
        x (int): number of occurrences.


    References:
        .. [#] Minitab (2019). Poisson Distribution. https://bityl.co/4uYc
        .. [#] Weisstein, Eric W. "Poisson Distribution." From MathWorld--A Wolfram Web Resource. https://mathworld.wolfram.com/PoissonDistribution.html.
        .. [#] Wikipedia contributors. (2020, December 16). Poisson distribution. In Wikipedia, The Free Encyclopedia. Retrieved 08:53, December 26, 2020, from https://en.wikipedia.org/w/index.php?title=Poisson_distribution&oldid=994605766
    """

    def __init__(self, λ:float):
        self.λ = λ

    def pmf(self, x: Union[List[int], int]) -> Union[float, _np.ndarray]:
        """
        Args:

            x (List[int]): random variable or list of random variables
            Reference: https://en.wikipedia.org/wiki/Poisson_distribution

        Returns: 
            probability mass evaluation of Poisson distribution to some point specified by the random variable
            or a list of its corresponding value specified by the parameter x.
        """

        def __generator(k, λ): 
            return (_np.power(λ, k) * _np.exp(-λ)) / _np.math.factorial(k)

        if isinstance(x, (List, _np.ndarray)):
            if any(type(i) is not int or i < 0 for i in x):
                raise TypeError('parameter x must be a positive integer')
            return _np.vectorize(__generator)(x, self.λ)

        if x < 0:
            raise ValueError('parameter x must be a positive integer')
        return __generator(x, self.λ)

    def cdf(self, x: Union[List[int], int]) -> Union[float, _np.ndarray]:
        """
        Args:

            x (List[int]): random variable or list of random variables
            Reference: https://en.wikipedia.org/wiki/Poisson_distribution

        Returns: 
            commulative density function of Poisson distribution to some point specified by the random variable
            or a list of its corresponding value specified by the parameter x.
        """
        λ = self.λ
        def __generator(k, λ): 
            return _gammainc(_floor(k + 1), λ) / _np.math.factorial(_floor(k))

        if isinstance(x, (List, _np.ndarray)):
            if any(type(i) is not int or i < 0 for i in x):
                raise TypeError('parameter x must be a positive integer')
            return _np.vectorize(__generator)(x, self.λ)

        if x < 0:
            raise ValueError('parameter x must be a positive integer')
        return __generator(x, self.λ)

    def mean(self) -> float:
        """
        Returns: 
            the mean of Poisson Distribution.
        """
        return self.λ

    def median(self) -> float:
        """
        Returns: 
            the median of Poisson Distribution.
        """
        λ = self.λ
        return λ + 1 / 3 - (0.02 / λ)

    def mode(self) -> Tuple[int, int]:
        """
        Returns: 
            the mode of Poisson Distribution.
        """
        λ = self.λ
        return _ceil(λ) - 1, _floor(λ)

    def var(self) -> float:
        """
        Returns: 
            the variance of Poisson Distribution.
        """
        return self.λ

    def skewness(self) -> float:
        """
        Returns: 
            the skewness of Poisson Distribution.
        """
        return pow(self.λ, -0.5)

    def kurtosis(self) -> float:
        """
        Returns: 
            the kurtosis of Poisson Distribution.
        """
        return 1/self.λ

    def summary(self) -> Dict[str, Union[float, int]]:
        """
        Summary statistic regarding the Poisson distribution which contains the following parts of the distribution:
        (mean, median, mode, var, std, skewness, kurtosis).

        Returns:
            Dict[str, Union[float, int]]
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }
