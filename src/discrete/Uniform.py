try:
    import numpy as np
    from typing import Union, Tuple, Dict
    from _base import Base
except Exception as e:
    print(f"some modules are missing {e}")


class Uniform(Base):
    """
    This contains methods for finding the probability mass function and 
    cumulative distribution function of Uniform distribution. Incudes scatter plot. 

    Args: 

        data (int): sample size

    Methods

        - pdf for evaluating or plotting probability mass function
        - cdf for evaluating or plotting cumulative distribution function
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - summary for printing the summary statistics of the distribution.
        - keys for returning a dictionary of summary statistics.

    Reference:
    - NIST/SEMATECH e-Handbook of Statistical Methods (2012). Uniform Distribution. Retrieved from http://www.itl.nist.gov/div898/handbook/, December 26, 2020.
    """

    def __init__(self, data):
        self.data = np.ones(data)

    def pmf(self, plot=False, xlim=None, ylim=None, xlabel=None, ylabel=None):
        """
        Args:

            plot (bool): returns scatter plot if true. 
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        Returns:
            either probability mass value of Uniform distribution or scatter plot
        """
        if plot == True:
            x = np.array([i for i in range(0, len(self.data))])
            y = np.array(
                [1 / len(self.data) for i in range(0, len(self.data))])
            return super().scatter(x, y, xlim, ylim, xlabel, ylabel)
        return 1 / len(self.data)

    def cdf(self,
            a,
            b,
            point=0,
            plot=False,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:

            a(int): lower limit of the distribution
            b(int): upper limit of the distribution
            point(int): point at which cumulative value is evaluated. Optional. 
            plot(bool): returns plot if true.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 


        Retruns:
            either cumulative distribution evaluation at some point or scatter plot.
        """

        def cdf_function(x, _a, _b): return (
            np.floor(x) - _a + 1) / (_b - _a + 1)
        if plot == True:
            x = np.array([i + 1 for i in range(a, b)])
            y = np.array([cdf_function(i, a, b) for i in x])
            return super().scatter(x, y, xlim, ylim, xlabel, ylabel)
        return cdf_function(point, a, b)

    def mean(self) -> float:
        """
        Returns the mean of Uniform Distribution.
        """
        return (self.a + self.b) / 2

    def median(self) -> float:
        """
        Returns the median of Uniform Distribution.
        """
        return (self.a + self.b) / 2

    def mode(self) -> Tuple[int, int]:
        """
        Returns the mode of Uniform Distribution.
        """
        return (self.a, self.b)

    def var(self) -> float:
        """
        Returns the variance of Uniform Distribution.
        """
        return (self.b - self.a)**2 / 12

    def skewness(self) -> int:
        """
        Returns the skewness of Uniform Distribution.
        """
        return 0

    def kurtosis(self) ->float:
        """
        Returns the kurtosis of Uniform Distribution.
        """
        return -6 / 5

    def summary(self, display=False) -> Union[None, Tuple[str, str, str, str, str, str, str]]:
        """
        Returns:  summary statistic regarding the Uniform-distribution which contains the following parts of the distribution:
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
        Summary statistic regarding the Uniform-distribution which contains the following parts of the distribution:
        (mean, median, mode, var, std, skewness, kurtosis).

        Returns:
            Dict[str, Union[float, int]]: [description]
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }
