try:
    from scipy.special import binom as _binom
    import numpy as np
    from math import sqrt as _sqrt, ceil as _ceil, floor as _floor
    from typing import Union, Tuple, Dict
    from _base import Base
except Exception as e:
    print(f"some modules are missing {e}")


class Binomial(Base):
    """
    This class contains functions for finding the probability mass function and 
    cumulative distribution function for binomial distirbution. 

    Args:

        n(int): number  of trials 
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
    - NIST/SEMATECH e-Handbook of Statistical Methods (2012). Binomial Distribution. 
    Retrieved at http://www.itl.nist.gov/div898/handbook/, December 26, 2000.
    - Wikipedia contributors. (2020, December 19). Binomial distribution. 
    In Wikipedia, The Free Encyclopedia. Retrieved 07:24, December 26, 2020, from https://en.wikipedia.org/w/index.php?title=Binomial_distribution&oldid=995095096
    - Weisstein, Eric W. "Binomial Distribution." From MathWorld--A Wolfram Web Resource. 
    https://mathworld.wolfram.com/BinomialDistribution.html
    """

    def __init__(self, n: int, p: Union[float, int], k: int):
        if type(n) and type(k) is not int:
            raise TypeError('parameters n and k must be of type int')

        if p < 0 or p > 1:
            raise ValueError('parameter p is constrained to ∈ [0,1]')

        self.n = n
        self.p = p
        self.k = k

    def pmf(self,
            interval=None,
            threshold=100,
            plot=False,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:

            interval(int): defaults to none. Only necessary for defining scatter plot.
            threshold(int): defaults to 100. Defines the sample points in scatter plot.
            plot(bool): if true, returns scatter plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 


        Returns: 
            either probability mass evaluation for some point or scatter plot of binomial distribution.
        """
        n = self.n
        p = self.p
        k = self.k

        def generator(n, p, k):
            def bin_coef(n, k): return _binom(n, k)  # assumes n>k
            if isinstance(k, list) == True:
                k_list = [i + 1 for i in range(0, len(k))]
                y = np.array([(bin_coef(n, k_) * pow(p, k_)) *
                              pow(1 - p, n - k_) for k_ in k_list])
                return y
            return (bin_coef(n, k) * pow(p, k)) * pow(1 - p, n - k)

        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = generator(n, p, x)
            return super().scatter(x, y, xlim, ylim, xlabel, ylabel)

        return generator(n, p, k)

    def cdf(self,
            interval=0,
            point=0,
            threshold=100,
            plot=False,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:

            interval(int): defaults to none. Only necessary for defining scatter plot.
            threshold(int): defaults to 100. Defines the sample points in scatter plot.
            plot(bool): if true, returns scatter plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 


        Returns: 
            either cumulative distirbution evaluation for some point or scatter plot of binomial distribution.
        """
        n = self.n
        p = self.p
        k = self.k

        def generator(n, p, k):
            def bin_coef(x): return np.array(
                (np.math.factorial(n) /
                 (np.math.factorial(x) * np.math.factorial(np.abs(n - x)))) *
                (pow(p, x) * pow((1 - p), n - x)))
            return np.cumsum([bin_coef(j) for j in range(0, k)], dtype=float)

        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = generator(n, p, len(x))
            return super().scatter(x, y, xlim, ylim, xlabel, ylabel)

        return generator(n, p, point)[
            point -
            1]  # will this output the cumulative sum at point requested?

    def mean(self) -> int:
        """
        Returns the mean of Binomial Distribution.
        """
        return self.n * self.p

    def median(self) -> Tuple[int, int]:
        """
        Returns the median of Binomial Distribution. Either one defined in the tuple of result.
        """
        n = self.n
        p = self.p
        return _floor(n * p), _ceil(n * p)

    def mode(self) -> Tuple[int, int]:
        """
        Returns the mode of Binomial Distribution. Either one defined in the tuple of result.
        """
        n = self.n
        p = self.p
        return _floor((n + 1) * p), _ceil((n + 1) * p) - 1

    def var(self) -> float:
        """
        Returns the variance of Binomial Distribution.
        """
        n = self.n
        p = self.p
        q = 1 - p
        return n * p * q

    def skewness(self) -> float:
        """
        Returns the skewness of Binomial Distribution.
        """
        n = self.n
        p = self.p
        q = 1 - p
        return (q - p) / _sqrt(n * p * q)

    def kurtosis(self) -> float:
        """
        Returns the kurtosis of Binomial Distribution.
        """
        n = self.n
        p = self.p
        q = 1 - p
        return (1 - 6 * p * q) / (n * p * q)

    def summary(self, display=False) -> Union[None, Tuple[str, str, str, str, str, str, str]]:
        """
        Returns:  summary statistic regarding the Binomial distribution which contains the following parts of the distribution:
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
        Summary statistic regarding the Binomial distribution which contains the following parts of the distribution:
        (mean, median, mode, var, std, skewness, kurtosis).

        Returns:
            Dict[str, Union[float, int]]: [description]
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }
