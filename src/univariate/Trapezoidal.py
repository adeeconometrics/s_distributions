try:
    import numpy as np
    from typing import Union, Tuple, Dict
    from _base import Base
except Exception as e:
    print(f"some modules are missing {e}")


class Trapezoidal(Base):
    """
    This class contains methods concerning Trapezoidal Distirbution.
    Args:

        a(float | a<d): lower bound
        b(float | a≤b<c): level start
        c(float | b<c≤d): level end
        d(float | c≤d): upper bound
        randvar(float | a≤randvar≤d): random variable

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

    Reference:
    - Wikipedia contributors. (2020, April 11). Trapezoidal distribution. In Wikipedia, The Free Encyclopedia.
    Retrieved 06:06, December 30, 2020, from https://en.wikipedia.org/w/index.php?title=Trapezoidal_distribution&oldid=950241388
    """

    def __init__(self, a: float, b: float, c: float, d: float, randvar: float):
        if a > d:
            raise ValueError(
                'lower bound(a) should be less than upper bound(d).')
        if a > b or b >= c:
            raise ValueError(
                'lower bound(a) should be less then or equal to level start (b) where (b) is less than level end(c).')
        if b >= c or c > d:
            raise ValueError(
                'level start(b) should be less then level end(c) where (c) is less then or equal to upper bound (d).')
        if c > d:
            raise ValueError(
                'level end(c) should be less than or equal to upper bound(d)')

        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.randvar = randvar

    def pdf(self,
            plot=False,
            interval=1,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None) -> Union[float, np.ndarray, None]:
        """
        Args:

            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true.
            xlabel(string): sets label in x axis. Only relevant when plot is true.
            ylabel(string): sets label in y axis. Only relevant when plot is true.


        Returns:
            either probability density evaluation for some point or plot of Trapezoidal distribution.
        """
        def __generator(a, b, c, d, x):
            if a <= x and x < b:
                return 2/(d+c-a-b) * (x-a)/(b-a)
            if b <= x and x < c:
                return 2/(d+c-a-b)
            if c <= x and x <= d:
                return (2/(d+c-a-b))*(d-x)/(d-c)

        if plot:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([__generator(self.a, self.b, self.c, self.d, i)
                         for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return __generator(self.a, self.b, self.c, self.d, self.randvar)

    def cdf(self,
            plot=False,
            interval=1,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None) -> Union[float, np.ndarray, None]:
        """
        Args:

            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true.
            xlabel(string): sets label in x axis. Only relevant when plot is true.
            ylabel(string): sets label in y axis. Only relevant when plot is true.


        Returns:
            either cumulative distribution evaluation for some point or plot of Trapezoidal distribution.
        """
        def __generator(a, b, c, d, x):
            if a <= x and x < b:
                return (x-a)**2/((b-a)*(d+c-a-b))
            if b <= x and x < c:
                return (2*x-a-b)/(d+c-a-b)
            if c <= x and x <= d:
                return 1 - (d-x)**2/((d+c-a-b)*(d-c))

        if plot:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([__generator(self.a, self.b, self.c, self.d, i)
                         for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return __generator(self.a, self.b, self.c, self.d, self.randvar)

    def pvalue(self) -> Union[float, str]:
        """
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Trapezoidal distribution evaluated at some random variable.
        """
        return "currently unsupported"

    def mean(self) -> float:
        """
        Returns: Mean of the Trapezoidal distribution.
        """
        a = self.a
        b = self.b
        c = self.c
        d = self.d

        return 1/(3*(d+c-b-a)) * ((d**3 - c**3)/(d-c) - (b**3 - a**3)/(b-a))

    def var(self) -> float:
        """
        Returns: Variance of the Trapezoidal distribution. Currently Unsupported.
        """
        a = self.a
        b = self.b
        c = self.c
        d = self.d

        mean = 1/(3*(d+c-b-a)) * ((d**3 - c**3)/(d-c) - (b**3 - a**3)/(b-a))
        return 1/(6*(d+c-b-a)) * ((d**4 - c**4)/(d-c) - (b**4 - a**4)/(b-a)) - pow(mean, 2)

    def summary(self, display=False) -> Union[None, Tuple[str, str, str, str, str, str, str]]:
        """
        Returns:  summary statistic regarding the Trapezoidal distribution which contains the following parts of the distribution:
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

    def keys(self) -> Dict[str, Union[float, Tuple[float]]]:
        """
        Summary statistic regarding the Trapezoidal distribution which contains the following parts of the distribution:
        (mean, median, mode, var, std, skewness, kurtosis).

        Returns:
            Dict[str, Union[float, Tuple[float]]]: [description]
        """
        return {
            'main': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.main(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }
