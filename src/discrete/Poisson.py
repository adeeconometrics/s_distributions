try:
    import numpy as np
    from scipy.special import gammainc as _gammainc
    from math import sqrt as _sqrt, ceil as _ceil, floor as _floor, log2 as _log2
    from typing import Union, Tuple, Dict
    from _base import Base
except Exception as e:
    print(f"some modules are missing {e}")


class Poisson(Base):
    """
    This class contains methods for evaluating some properties of the poisson distribution. 
    As lambda increases to sufficiently large values, the normal distribution (λ, λ) may be used to 
    approximate the Poisson distribution.

    Use the Poisson distribution to describe the number of times an event occurs in a finite observation space.


    Args: 

        λ(float): expected rate if occurrences.
        k(int): number of occurrences.

    Methods:

        - pmf for probability mass function.
        - cdf for cumulative distribution function.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - summary for printing the summary statistics of the distribution.
        - keys for returning a dictionary of summary statistics.

    References:
        -  Minitab (2019). Poisson Distribution. https://bityl.co/4uYc
        - Weisstein, Eric W. "Poisson Distribution." From MathWorld--A Wolfram Web Resource. 
        https://mathworld.wolfram.com/PoissonDistribution.html
        - Wikipedia contributors. (2020, December 16). Poisson distribution. In Wikipedia, The Free Encyclopedia.
         Retrieved 08:53, December 26, 2020, from https://en.wikipedia.org/w/index.php?title=Poisson_distribution&oldid=994605766
    """

    def __init__(self, λ: Union[int, float], k: int):
        if type(k) is not int:
            raise TypeError('parameter k should be of type int')

        self.k = k
        self.λ = λ

    def pmf(self, x:List[int] = None) -> Union[int, float, List[int]]:
        """
        Args:

            x (List[int]): random variable or list of random variables
            Reference: https://en.wikipedia.org/wiki/Poisson_distribution

        Returns: 
            probability mass evaluation of Poisson distribution to some point specified by the random variable
            or a list of its corresponding value specified by the parameter x.
        """

        k = self.k
        λ = self.λ

        def __generator(k, λ): return (pow(λ, k) * np.exp(-λ)
                                     ) / np.math.factorial(k)

        if x is not None and issubclass(x, List):
            return [__generator(p, i) for i in x]

        return __generator(k, λ)

    def cdf(self, x:List[int] = None) -> Union[int, float, List[int]]:
        """
        Args:

            x (List[int]): random variable or list of random variables
            Reference: https://en.wikipedia.org/wiki/Poisson_distribution

        Returns: 
            commulative density function of Poisson distribution to some point specified by the random variable
            or a list of its corresponding value specified by the parameter x.
        """
        k = self.k
        λ = self.λ
        def __generator(k, λ): return _gammainc(_floor(k + 1), λ
                                              ) / np.math.factorial(_floor(k))
        if x is not None and issubclass(x, List):
            return [__generator(p, i) for i in x]
            
        return __generator(k, λ)

    def mean(self) -> float:
        """
        Returns the mean of Poisson Distribution.
        """
        return self.λ

    def median(self) -> float:
        """
        Returns the median of Poisson Distribution.
        """
        λ = self.λ
        return λ + 1 / 3 - (0.02 / λ)

    def mode(self) -> Tuple[int, int]:
        """
        Returns the mode of Poisson Distribution.
        """
        λ = self.λ
        return _ceil(λ) - 1, _floor(λ)

    def var(self) -> float:
        """
        Returns the variance of Poisson Distribution.
        """
        return self.λ

    def skewness(self) -> float:
        """
        Returns the skewness of Poisson Distribution.
        """
        return pow(self.λ, -1 / 2)

    def kurtosis(self) -> float:
        """
        Returns the kurtosis of Poisson Distribution.
        """
        return pow(self.λ, -1)

    def summary(self, display=False) -> Union[None, Tuple[str, str, str, str, str, str, str]]:
        """
        Returns:  summary statistic regarding the Poisson distribution which contains the following parts of the distribution:
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
        Summary statistic regarding the Poisson distribution which contains the following parts of the distribution:
        (mean, median, mode, var, std, skewness, kurtosis).

        Returns:
            Dict[str, Union[float, int]]: [description]
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }
