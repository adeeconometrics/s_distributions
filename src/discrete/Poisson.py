try:
    import numpy as np
    from scipy.special import gammainc as _gammainc
    from math import sqrt as _sqrt, ceil as _ceil, floor as _floor, log2 as _log2
    from typing import Union, Typing, Dict
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
        - print_summary for printing summary statistics.

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

            Reference: https://en.wikipedia.org/wiki/Poisson_distribution

        Returns: 
            either probability mass evaluation for some point or scatter plot of poisson distribution.
        """
        k = self.k
        λ = self.λ

        def generator(k, λ): return (pow(λ, k) * np.exp(-λ)
                                     ) / np.math.factorial(k)
        if plot == True:
            x = np.linspace(1, interval, threshold)
            y = np.array([generator(x_temp, λ) for x_temp in x])
            return super().scatter(x, y, xlim, ylim, xlabel, ylabel)
        return generator(k, λ)

    def cdf(self,
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

            Reference: https://en.wikipedia.org/wiki/Poisson_distribution

        Returns: 
            either cumulative distribution evaluation for some point or scatter plot of poisson distribution.
        """
        k = self.k
        λ = self.λ
        def generator(k, λ): return _gammainc(np.floor(k + 1), λ
                                              ) / np.math.factorial(np.floor(k))
        if plot == True:
            x = np.linspace(1, interval, threshold)
            y = np.array([generator(x_temp, λ) for x_temp in x])
            return super().scatter(x, y, xlim, ylim, xlabel, ylabel)
        return generator(k, λ)

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
            'main': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.main(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }
