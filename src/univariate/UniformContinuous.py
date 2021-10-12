try:
    import numpy as np
    from math import sqrt as _sqrt, log as _log
    from typing import Union, Tuple, Dict
    from . import Base
except Exception as e:
    print(f"some modules are missing {e}")


class Uniform(Base):
    """
    This class contains methods concerning the Continuous Uniform Distribution.

    Args:

        a(int): lower limit of the distribution
        b(int): upper limit of the distribution

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

    Referene:
    - Weisstein, Eric W. "Uniform Distribution." From MathWorld--A Wolfram Web Resource.
    https://mathworld.wolfram.com/UniformDistribution.html
    """

    def __init__(self, a: int, b: int) -> None:
        if type(a) and type(b) is int:
            raise TypeError('parameters a, b must be of type int.')
        self.a = a
        self.b = b

    def pdf(self, plot=False, xlim=None, ylim=None, xlabel=None, ylabel=None) -> Union[number, np.ndarray, None]:
        """
        Args:

            plot (bool): returns plot if true.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true.
            xlabel(string): sets label in x axis. Only relevant when plot is true.
            ylabel(string): sets label in y axis. Only relevant when plot is true.

        Returns:
            either plot of the distribution or probability density evaluation at a to b.
        """
        a = self.a
        b = self.b
        threshold = b - a

        def __generator(a, b, x): return 1 / \
            (b - a) if a <= x and x <= b else 0
        if plot:
            x = np.linspace(a, b, threshold)
            y = np.array([__generator(a, b, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return __generator(a, b, abs(b - a))

    def cdf(self, plot=False, xlim=None, ylim=None, xlabel=None, ylabel=None) -> Union[number, np.ndarray, None]:
        """
        Args:

            plot (bool): returns plot if true.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true.
            xlabel(string): sets label in x axis. Only relevant when plot is true.
            ylabel(string): sets label in y axis. Only relevant when plot is true.

        Returns:
            either plot of the distribution or probability density evaluation at a to b.
        """
        a = self.a
        b = self.b
        threshold = b - a

        def __generator(a, b, x):
            if x < a:
                return 0
            if (a <= x and x <= b):
                return (x - a) / (b - a)
            if x > b:
                return 1

        if plot:
            x = np.linspace(a, b, threshold)
            y = np.array([__generator(a, b, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return __generator(a, b, threshold)  # what does it really say?

    def mean(self) -> float:
        """
        Returns: Mean of the Uniform distribution.
        """
        return 1 / 2 * (self.a + self.b)

    def median(self) -> float:
        """
        Returns: Median of the Uniform distribution.
        """
        return 1 / 2 * (self.a + self.b)

    def mode(self) -> Tuple[int, int]:
        """
        Returns: Mode of the Uniform distribution.

        Note that the mode is any value in (a,b)
        """
        return (self.a, self.b)

    def var(self) -> float:
        """
        Returns: Variance of the Uniform distribution.
        """
        return 1 / 12 * pow(self.b - self.a,2)

    def std(self) -> float:
        """
        Returns: Standard deviation of the Uniform distribution.
        """
        return _sqrt(1 / 12 * pow(self.b - self.a,2))

    def skewness(self) -> int:
        """
        Returns: Skewness of the Uniform distribution.
        """
        return 0

    def kurtosis(self) -> float:
        """
        Returns: Kurtosis of the Uniform distribution.
        """
        return -6 / 5

    def entropy(self) -> float:
        """
        Returns: entropy of uniform Distirbution.

        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier.
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        return _log(self.b-self-a)

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
            'main': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.main(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }
