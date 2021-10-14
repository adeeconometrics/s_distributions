try:
    from scipy.special import binom as _binom
    import numpy as np
    from math import sqrt as _sqrt, ceil as _ceil, floor as _floor
    from typing import Union, Typing, Dict
    from _base import Base
except Exception as e:
    print(f"some modules are missing {e}")


class Hypergeometric(Base):
    """
    This class contains methods concerning pmf and cdf evaluation of the hypergeometric distribution. 
    Describes the probability if k successes (random draws for which the objsect drawn has specified deature)
    in n draws, without replacement, from a finite population size N that contains exactly K objects with that
    feature, wherein each draw is either a success or a failure. 

    Args:

        N(int): population size
        K(int): number of success states in the population
        k(int): number of observed successes
        n(int): number of draws 

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
    - Weisstein, Eric W. "Hypergeometric Distribution." From MathWorld--A Wolfram Web Resource. 
    https://mathworld.wolfram.com/HypergeometricDistribution.html
    - Wikipedia contributors. (2020, December 22). Hypergeometric distribution. In Wikipedia, The Free Encyclopedia. 
    Retrieved 08:38, December 26, 2020, from https://en.wikipedia.org/w/index.php?title=Hypergeometric_distribution&oldid=995715954

    """

    def __init__(self, N: int, K: int, k: int, n: int):
        if type(N) and type(n) and type(K) and type(k) is not int:
            raise TypeError('all parameters must be of type int')

        self.N = N
        self.K = K
        self.k = k
        self.n = n

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
            either probability mass evaluation for some point or scatter plot of hypergeometric distribution.
        """
        n = self.n
        k = self.k
        N = self.N
        K = self.K

        def generator(N, n, K, k): return (_binom(n, k) * _binom(
            N - K, n - k)) / _binom(N, n)  # assumes n>k

        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array(
                [generator(N, n, K, x_temp) for x_temp in range(0, len(x))])
            return super().scatter(x, y, xlim, ylim, xlabel, ylabel)

        return generator(N, n, K, k)

    def cdf(self,
            interval=None,
            threshold=100,
            plot=False,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):  # np.cumsum()
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
            either cumulative distribution evaluation for some point or scatter plot of hypergeometric distribution.
        """
        n = self.n
        k = self.k
        N = self.N
        K = self.K

        def generator(N, n, K, k): return (_binom(n, k) * _binom(
            N - K, n - k)) / _binom(N, n)  # assumes n>k

        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.cumsum(
                [generator(N, n, K, x_temp) for x_temp in range(0, len(x))])
            return super().scatter(x, y, xlim, ylim, xlabel, ylabel)
        return np.cumsum(generator(N, n, K, k))[k - 1]

    def mean(self) -> float:
        """
        Returns the mean of Hypergeometric Distribution.
        """
        return self.n * (self.K / self.N)

    def median(self) -> str:
        """
        Returns the median of Hypergeometric Distribution. Currently unsupported or undefined.
        """
        return "undefined"

    def mode(self) -> Tuple[int, int]:
        """
        Returns the mode of Hypergeometric Distribution.
        """
        n = self.n
        N = self.N
        k = self.k
        K = self.K
        return _ceil(((n + 1) * (K + 1)) / (N + 2)) - 1, _floor(
            ((n + 1) * (K + 1)) / (N + 2))

    def var(self) -> float:
        """
        Returns the variance of Hypergeometric Distribution.
        """
        n = self.n
        N = self.N
        k = self.k
        K = self.K
        return n * (K / N) * ((N - K) / N) * ((N - n) / (N - 1))

    def skewness(self) -> float:
        """
        Returns the skewness of Hypergeometric Distribution.
        """
        n = self.n
        N = self.N
        k = self.k
        K = self.K
        return ((N - 2 * K) * pow(N - 1, 1 / 2) *
                (N - 2 * n)) / (_sqrt(n * K * (N - K) * (N - n)) * (N - 2))

    def kurtosis(self) -> float:
        """
        Returns the kurtosis of Hypergeometric Distribution.
        """
        n = self.n
        N = self.N
        k = self.k
        K = self.K
        scale = 1 / (n * k(N - K) * (N - n) * (N - 2) * (N - 3))
        return scale * ((N - 1) * N**2 * (N * (N + 1) - (6 * K * (N - K)) -
                                          (6 * n * (N - n))) +
                        (6 * n * K(N - K) * (N - n) * (5 * N - 6)))

    def summary(self, display=False) -> Union[None, Tuple[str, str, str, str, str, str, str]]:
        """
        Returns:  summary statistic regarding the Hypergeometric distribution which contains the following parts of the distribution:
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
        Summary statistic regarding the Hypergeometric distribution which contains the following parts of the distribution:
        (mean, median, mode, var, std, skewness, kurtosis).

        Returns:
            Dict[str, Union[float, int]]: [description]
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }
