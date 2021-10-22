try:
    import numpy as _np
    from typing import Union, Tuple, Dict, List
    from math import sqrt as _sqrt, log as _log, exp as _exp
    from univariate._base import SemiInfinite
except Exception as e:
    print(f"some modules are missing {e}")


class Explonential(SemiInfinite):
    """
    This class contans methods for evaluating Exponential Distirbution.

    Args:

        - lambda_(float | x>0): rate parameter.
        - x(float | x>0): random variable.

    Methods:

        - pdf for probability density function.
        - cdf for cumulative distribution function.
        - pvalue for p-values.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - std for evaluating the standard deviation of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - entropy for differential entropy of the distribution.
        - summary for printing the summary statistics of the distribution.
        - keys for returning a dictionary of summary statistics.

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
            return 0

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
            return 0

        if x is not None:
            if not isinstance(x, (_np.ndarray, List)):
                raise TypeError(f'parameter x only accepts List types or numpy.ndarray')
            else:
                return [__generator(lambda_, i) for i in x]

        return __generator(lambda_, self.x)

    def pvalue(self, x_lower=0, x_upper=None) -> Optional[float]:
        """
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Exponential distribution evaluated at some random variable.
        """
        lambda_ = self.lambda_
        x = self.x
        if x_lower < 0:
            raise Exception(
                'x_lower cannot be lower than 0. Entered value: {}'.format(x_lower))
        if x_upper is None:
            x_upper = x

        def __cdf(x, lambda_):
            if x > 0:
                return 1 - _exp(-lambda_ * x)
            return 0
        return __cdf(x_upper, lambda_) - __cdf(x_lower, lambda_)

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

    def mode(self) -> int:
        """
        Returns: Mode of the Exponential distribution
        """
        return 0

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

    def skewness(self) -> int:
        """
        Returns: Skewness of the Exponential distribution
        """
        return 2

    def kurtosis(self) -> int:
        """
        Returns: Kurtosis of the Exponential distribution
        """
        return 6

    def entorpy(self) -> float:
        """
        Returns: differential entropy of the Exponential distribution

        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier.
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        return 1 - _log(self.lambda_)

    def summary(self, display=False) -> Union[None, Tuple[str, str, str, str, str, str, str]]:
        """
        Returns:  summary statistic regarding the Exponential distribution which contains the following parts of the distribution:
                (mean, median, mode, var, std, skewness, kurtosis). If the display parameter is True, the function returns None
                and prints out the summary of the distribution. 
        """
        if display == True:
            cstr = " summary statistics "
            print(cstr.center(40, "="))
            print(f"mean: {self.mean()}", f"median: {self.median()}",
                  f"mode: {self.mode()}", f"var: {self.var()}", f"std: {self.std()}",
                  f"skewness: {self.skewness()}", f"kurtosis: {self.kurtosis()}", sep='\n')

            return None
        else:
            return (f"mean: {self.mean()}", f"median: {self.median()}",
                    f"mode: {self.mode()}", f"var: {self.var()}", f"std: {self.std()}",
                    f"skewness: {self.skewness()}", f"kurtosis: {self.kurtosis()}")

    def keys(self) -> Dict[str, Union[float, int]]:
        """
        Summary statistic regarding the Exponential-distribution which contains the following parts of the distribution:
        (mean, median, mode, var, std, skewness, kurtosis).

        Returns:
            Dict[str, Union[float, int]]: [description]
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }
