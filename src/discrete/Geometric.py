try:
    import numpy as np
    from math import sqrt as _sqrt, ceil as _ceil, floor as _floor, log2 as _log2
    from typing import Union, Tuple, Dict, List
    from _base import Base
except Exception as e:
    print(f"some modules are missing {e}")

class Geometric(Base):
    """
    This class contains functions for finding the probability mass function and 
    cumulative distribution function for geometric distribution. We consider two definitions 
    of the geometric distribution: one concerns itself to the number of X of Bernoulli trials
    needed to get one success, supported on the set {1,2,3,...}. The second one concerns with 
    Y=X-1 of failures before the first success, supported on the set {0,1,2,3,...}. 

    Args:

        p(float ∈ [0,1]): success probability for each trial
        k(int): number of successes 

    Methods: 

        - pmf for probability mass function.
        - cdf for cumulative distribution function.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - print_summary for printing summary statistics.

    References:
    - Weisstein, Eric W. "Geometric Distribution." From MathWorld--A Wolfram Web Resource. https://mathworld.wolfram.com/GeometricDistribution.html
    - Wikipedia contributors. (2020, December 27). Geometric distribution. In Wikipedia, The Free Encyclopedia. 
    Retrieved 12:05, December 27, 2020, from https://en.wikipedia.org/w/index.php?title=Geometric_distribution&oldid=996517676
    """
    def __init__(self, p, k):
        if type(k) is not int:
            raise TypeError('parameter k must be of type int')
        if p < 0 or p > 1:
            raise ValueError('parameter p is constrained at')

        self.p = p
        self.k = k

    def pmf(self, x:List[int] = None, _type:str = 'first') -> Union[List[int], int, float]:
        """
        Args:         
            type (keyvalue ∈[fist, second]): defaults to first. Reconfigures the type of distribution.
            x (List[int]): list of random variables

            Reference: https://en.wikipedia.org/wiki/Geometric_distribution

        Returns: 
            probability mass evaluation of geometric distribution to some point specified by k or a 
            list of its corresponding value specified by the parameter x.
            
        Note: there are two configurations of pmf. 
        """
        p = self.p
        k = self.k
        if _type == "first":
            __generator = lambda p, k: pow(1 - p, k - 1) * p
        elif _type == "second":
            __generator = lambda p, k: pow(1 - p, k) * p
        else:  
            raise ValueError("Invalid argument. Type is either 'first' or 'second'.")

        if x is not None and issubclass(x, List):
            return [__generator(p, k_i) for k_i in x]

        return __generator(p, k)

    def cdf(self, x:List[int] = None, _type:str = 'first') -> Union[List[int], int, float]:
        """
        Args: 
     
            type(keyvalue ∈[fist, second]): defaults to first. Reconfigures the type of distribution.


            for context see: https://en.wikipedia.org/wiki/Geometric_distribution

        Returns: 
            cumulative distribution evaluation to some point specified by k or scatter plot of geometric distribution.
            
        Note: there are two configurations of cdf. 
        """
        p = self.p
        k = self.k
        if _type == "first":
            __generator = lambda p, k: 1 - pow(1 - p, k)
        elif _type == "second":
            __generator = lambda p, k: 1 - pow(1 - p, k + 1)
        else:  # supposed to raise exception when failed
            return print(
                "Invalid argument. Type is either 'first' or 'second'.")

        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([__generator(p, k_i) for k_i in x])
            return super().scatter(x, y, xlim, ylim, xlabel, ylabel)

        return __generator(p, k)

    def mean(self, _type='first') -> Union[int, str]:
        """
        Args:

            type(string): defaults to first type. Valid types: "first", "second".
        Returns the mean of Geometric Distribution.
        """
        p = self.p
        if _type == "first":
            return 1 / p
        elif _type == "second":
            return (1 - p) / p
        else:  # supposed to raise exception when failed
            return print(
                "Invalid argument. Type is either 'first' or 'second'.")

    def median(self, _type = 'first') -> Union[int, str]:
        """
        Args:

            type(string): defaults to first type. Valid types: "first", "second".
        Returns the median of Geometric Distribution.
        """
        if type == "first":
            return _ceil(1 / (_log2(1 - self.p)))
        elif type == "second":
            return _ceil(1 / (_log2(1 - self.p))) - 1
        else:  # supposed to raise exception when failed
            return print(
                "Invalid argument. Type is either 'first' or 'second'.")

    def mode(self, _type:str = 'first') -> Union[int, str]:
        """
        Args:

            type(string): defaults to first type. Valid types: "first", "second".
        Returns the mode of Geometric Distribution.
        """
        if _type == "first":
            return 1
        elif _type == "second":
            return 0
        else:  # supposed to raise exception when failed
            return print(
                "Invalid argument. Type is either 'first' or 'second'.")

    def var(self) -> float:
        """
        Returns the variance of Geometric Distribution.
        """
        return (1 - self.p) / self.p**2

    def skewness(self) -> float:
        """
        Returns the skewness of Geometric Distribution.
        """
        return (2 - self.p) / _sqrt(1 - self.p)

    def kurtosis(self) -> float:
        """
        Returns the kurtosis of Geometric Distribution.
        """
        return 6 + (self.p**2 / (1 - self.p))

    def summary(self, display=False) -> Union[None, Tuple[str, str, str, str, str, str, str]]:
        """
        Returns:  summary statistic regarding the Geometric distribution which contains the following parts of the distribution:
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
        Summary statistic regarding the Geometric distribution which contains the following parts of the distribution:
        (mean, median, mode, var, std, skewness, kurtosis).

        Returns:
            Dict[str, Union[float, int]]: [description]
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }

