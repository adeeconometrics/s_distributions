try:
    import numpy as _np
    from typing import Union, Tuple, Dict, List
    from math import sqrt as _sqrt, log as _log, exp as _exp
    from univariate._base import SemiInfinite
except Exception as e:
    print(f"some modules are missing {e}")


class Exponential(SemiInfinite):
    """
    This class contans methods for evaluating Exponential Distirbution.

    Args:

        - lambda_(float | x>0): rate parameter.
        - x(float | x>0): random variable.

    References:
    - Weisstein, Eric W. "Exponential Distribution." From MathWorld--A Wolfram Web Resource.
    https://mathworld.wolfram.com/ExponentialDistribution.html
    - Wikipedia contributors. (2020, December 17). Exponential distribution. In Wikipedia, The Free Encyclopedia.
    Retrieved 04:38, December 23, 2020, from https://en.wikipedia.org/w/index.php?title=Exponential_distribution&oldid=994779060
    """

    def __init__(self, lambda_: float, x: float = 1.0):
        if lambda_ < 0:
            raise ValueError(
                f'lambda parameter should be greater than 0. Entered value for lambda_:{lambda_}')
        if x < 0:
            raise ValueError(
                f'random variable should be greater than 0. Entered value for x:{x}')

        self.lambda_ = lambda_
        self.x = x

    def pdf(self, x: Union[List[float], _np.ndarray] = None) -> Union[float, List]:
        """
        Args:

            x (List[float], numpy.ndarray): random variable or list of random variables

        Returns:
            either cumulative distribution evaluation for some point or plot of Exponential distribution.
        """
        lambda_ = self.lambda_

        def __generator(lambda_:float, x:float) -> float:
            if x >= 0:
                return lambda_ * _exp(-(lambda_ * x))
            return 0.0

        if x is not None:
            if not isinstance(x, (_np.ndarray, List)):
                raise TypeError(f'parameter x only accepts List types or numpy.ndarray')
            else:
                return [__generator(lambda_, i) for i in x]
                
        return __generator(lambda_, self.x)

    def cdf(self,x: Union[List[float], _np.ndarray] = None) -> Union[float, List]:
        """
        Args:

            x (List[float], numpy.ndarray): random variable or list of random variables

        Returns:
            either comulative distribution evaluation for some point or plot of Exponential distribution.
        """ 
        lambda_ = self.lambda_

        def __generator(lambda_:float, x:float) -> float:
            if x > 0:
                return 1 - _exp(-lambda_ * x)
            return 0.0

        if x is not None:
            if not isinstance(x, (_np.ndarray, List)):
                raise TypeError(f'parameter x only accepts List types or numpy.ndarray')
            else:
                return [__generator(lambda_, i) for i in x]

        return __generator(lambda_, self.x)

    def mean(self) -> float:
        """
        Returns: Mean of the Exponential distribution
        """
        return 1 / self.lambda_

    def median(self) -> float:
        """
        Returns: Median of the Exponential distribution
        """
        return _log(2) / self.lambda_

    def mode(self) -> float:
        """
        Returns: Mode of the Exponential distribution
        """
        return 0.0

    def var(self) -> float:
        """
        Returns: Variance of the Exponential distribution
        """
        return 1 / pow(self.lambda_, 2)

    def std(self) -> float:
        """
        Returns: Standard deviation of the Exponential distribution
        """
        return _sqrt(self.var())

    def skewness(self) -> float:
        """
        Returns: Skewness of the Exponential distribution
        """
        return 2.0

    def kurtosis(self) -> float:
        """
        Returns: Kurtosis of the Exponential distribution
        """
        return 6.0

    def entorpy(self) -> float:
        """
        Returns: differential entropy of the Exponential distribution

        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier.
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        return 1 - _log(self.lambda_)

    def summary(self) -> Dict[str, Union[float, int]]:
        """
        Summary statistic regarding the Exponential-distribution which contains the following parts of the distribution:
        (mean, median, mode, var, std, skewness, kurtosis).

        Returns:
            Dict[str, Union[float, int]]
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }
