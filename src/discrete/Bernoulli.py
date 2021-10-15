try:
    from math import sqrt as _sqrt
    from typing import Union, Tuple, Dict, List
    from ._base import Base
except Exception as e:
    print(f"some modules are missing {e}")


class Bernoulli(Base):
    """
    This class contains methods concerning the Bernoulli Distribution. Bernoulli Distirbution is a special
    case of Binomial Distirbution. 
    Args:

        - p(int): event of success. 
        - k(float ∈[0,1]): possible outcomes
    Methods:

        - pmf for evaluating or list for plotting probability mass function
        - cdf for evaluating or list for plotting cumulative distribution function
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - print_summary for printing summary statistics.

    References:
        - Weisstein, Eric W. "Bernoulli Distribution." From MathWorld--A Wolfram Web Resource. 
        https://mathworld.wolfram.com/BernoulliDistribution.html
        - Wikipedia contributors. (2020, December 26). Bernoulli distribution. In Wikipedia, The Free Encyclopedia. 
        Retrieved 10:18, December 26, 2020, from https://en.wikipedia.org/w/index.php?title=Bernoulli_distribution&oldid=996380822
    """

    def __init__(self, p: int, k: float):
        if type(p) is not int:
            raise TypeError('parameter p must be of type int')
        if k < 0 or k > 1:
            raise ValueError('parameter k is constrained in ∈ [0,1]')

        self.p = p
        self.k = k

    def pmf(self, x:List[int] = None) -> Union[int, float, List[int]]:
        """
        Args:

            x (List[int]): random variable or list of random variables

        Returns: 
            probability mass evaluation of Bernoulli distribution to some point specified by the random variable
            or a list of its corresponding value specified by the parameter x.
        """
        p = self.p
        k = self.k

        def __generator(p, k): return p**k * pow(1 - p, 1 - k)

        if x is not None and issubclass(x, List):
            return [__generator(p, i) for i in x]

        return __generator(p, x)

    def cdf(self, x:List[int] = None):
        """
        Args:

            x (List[int]): list of random variables

        Returns: 
            commulative density function of Bernoulli distribution to some point specified by the random variable
            or a list of its corresponding value specified by the parameter x.
        """
        p = self.p
        k = self.k

        def __generator(k, p):
            if k < 0:
                return 0
            elif k >= 0 and k < 1:
                return 1 - p
            elif k >= 1:
                return 1

        if x is not None and issubclass(x, List):
            return [__generator(p, i) for i in x]

        return __generator(p, k)

    def mean(self) -> int:
        """
        Returns the mean of Bernoulli Distribution.
        """
        return self.p

    def median(self) -> Union[Tuple[int, int], int]:
        """
        Returns the median of Bernoulli Distribution.
        """
        p = self.p
        if p < 1 / 2:
            return 0
        if p == 1 / 2:
            return (0, 1)
        if p > 1 / 2:
            return 1

    def mode(self) -> Union[Tuple[int, int], int]:
        """
        Returns the mode of Bernoulli Distribution.
        """
        p = self.p
        if p < 1 / 2:
            return 0
        if p == 1 / 2:
            return (0, 1)
        if p > 1 / 2:
            return 1

    def var(self) -> float:
        """
        Returns the variance of Bernoulli Distribution.
        """
        p = self.p
        q = 1 - p
        return p * q

    def std(self) -> float:
        """
        Returns the variance of Bernoulli Distribution.
        """
        p = self.p
        q = 1 - p
        return _sqrt(p * q)

    def skewness(self) -> float:
        """
        Returns the skewness of Bernoulli Distribution.
        """
        p = self.p
        q = 1 - p
        return (q - p) / _sqrt(p * q)

    def kurtosis(self) -> float:
        """
        Returns the kurtosis of Bernoulli Distribution.
        """
        p = self.p
        q = 1 - p
        return (1 - 6 * p * q) / (p * q)

    def summary(self, display=False) -> Union[None, Tuple[str, str, str, str, str, str, str]]:
        """
        Returns:  summary statistic regarding the Bernoulli distribution which contains the following parts of the distribution:
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
        Summary statistic regarding the Bernoulli distribution which contains the following parts of the distribution:
        (mean, median, mode, var, std, skewness, kurtosis).

        Returns:
            Dict[str, Union[float, int]]: [description]
        """
        return {
            'main': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }
