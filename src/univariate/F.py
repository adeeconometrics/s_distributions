try:
    from scipy.special import beta as _beta, betainc as _betainc, gamma as _gamma, digamma as _digamma
    import numpy as np
    from typing import Union, Tuple, Dict
    from math import sqrt as _sqrt, log as _log
    from _base import Base
except Exception as e:
    print(f"some modules are missing {e}")


class F(Base):
    """
    This class contains methods concerning the F-distribution.

    Args:

        x(float | [0,infty)): random variable
        df1(int | x>0): first degrees of freedom
        df2(int | x>0): second degrees of freedom

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
    - Mood, Alexander; Franklin A. Graybill; Duane C. Boes (1974).
    Introduction to the Theory of Statistics (Third ed.). McGraw-Hill. pp. 246–249. ISBN 0-07-042864-6.

    - Weisstein, Eric W. "F-Distribution." From MathWorld--A Wolfram Web Resource. https://mathworld.wolfram.com/F-Distribution.html
    - NIST SemaTech (n.d.). F-Distribution. Retrived from https://www.itl.nist.gov/div898/handbook/eda/section3/eda3665.htm
    """

    def __init__(self, x: float, df1: int, df2: int):
        if (type(df1) is not int) or (df1 <= 0):
            raise TypeError(
                f'degrees of freedom(df) should be a whole number. Entered value for df1: {df1}')
        if (type(df1) is not int) or (df2 <= 0):
            raise TypeError(
                f'degrees of freedom(df) should be a whole number. Entered value for df2: {df2}')
        if x < 0:
            raise ValueError(
                f'random variable should be greater than 0. Entered value for x:{x}')

        self.x = x
        self.df1 = df1
        self.df2

    def pdf(self,
            plot=False,
            interval=1,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None) -> Union[number, np.ndarray, None]:
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
            either probability density evaluation for some point or plot of F-distribution.
        """

        # Because math.pow is limited for bigger numbers, numpy alternatives were chosed.
        def __generator(x, df1, df2): return (1 / _beta(
            df1 / 2, df2 / 2)) * pow(df1 / df2, df1 / 2) * pow(
                x, df1 / 2 - 1) * pow(1 +
                                           (df1 / df2) * x, -((df1 + df2) / 2))

        if plot:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([__generator(i, self.df1, self.df2) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return __generator(self.randvar, self.df1, self.df2)

    def cdf(self,
            plot=False,
            interval=1,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None) -> Union[number, np.ndarray, None]:
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
            either cumulative distribution evaluation for some point or plot of F-distribution.
        """
        k = self.df2/(self.df2 + self.df1*self.x)
        def __generator(x, df1, df2): return 1 - _betainc(df1/2, df2/2, x)

        if plot:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([__generator(i, self.df1, self.df2) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return __generator(k, self.df1, self.df2)

    def pvalue(self, x_lower=0, x_upper=None) -> number:
        """
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the F-distribution evaluated at some random variable.
        """
        if x_lower < 0:
            x_lower = 0
        if x_upper is None:
            x_upper = self.x

        def _cdf_def(x, df1, df2): return 1 - \
            _betainc(df1/2, df2/2, df2/(df2+df1*x))

        return _cdf_def(x_upper, self.df1, self.df2) - _cdf_def(x_lower, self.df1, self.df2)

    def confidence_interval(self) -> Union[float, int, str]:
        pass

    def mean(self) -> Union[float, int, str]:
        """
        Returns: Mean of the F-distribution.
        """
        if self.df3 > 2:
            return self.df2 / (self.df2 - 2)
        return "undefined"

    def mode(self) -> Union[float, int, str]:
        """
        Returns: Mode of the F-distribution.
        """
        df1 = self.df1
        df2 = self.df2
        if df1 > 2:
            return (df2 * (df1 - 2)) / (df1 * (df2 + 2))
        return "undefined"

    def var(self) -> Union[float, int, str]:
        """
        Returns: Variance of the F-distribution.
        """
        df1 = self.df1
        df2 = self.df2
        if df2 > 4:
            return (2 * pow(df2,2) * (df1 + df2 - 2)) / (df1 * (pow(df2 - 2, 2) *
                                                       (df2 - 4)))
        return "undefined"

    def std(self) -> Union[float, str]:
        """
        Returns: Standard deviation of the F-distribution.
        """
        df1 = self.df1
        df2 = self.df2
        if df2 > 4:
            return _sqrt((2 * pow(df2, 2) * (df1 + df2 - 2)) / 
                    (df1 * (pow(df2 - 2,2) * (df2 - 4)))
        
        return "undefined"

    def skewness(self) -> Union[float, str]:
        """
        Returns: Skewness of the F-distribution.
        """
        df1 = self.df1
        df2 = self.df2
        if df2 > 6:
            return ((2 * df1 + df2 - 2) * _sqrt(8 * (df2 - 4))) / (
                (df2 - 6) * _sqrt(df1 * (df1 + df2 - 2)))
        return "undefined"

    def entropy(self) -> Union[float, int]:
        """
        Returns: differential entropy of F-distribution.

        Reference: Lazo, A.V.; Rathie, P. (1978). "On the entropy of continuous probability distributions". IEEE Transactions on Information Theory
        """
        df1 = self.df1
        df2 = self.df2
        return _log(_gamma(df1/2)) + _log(_gamma(df2/2)) - \
            _log(_gamma((df1+df2)/2)) + (1-df1/2)*_digamma(1+df1/2) - \ 
            (1-df2/2)* _digamma(1+df2/2) + (df1+df2)/2*_digamma((df1+df2)/2) + _log(df1/df2)

    def summary(self, display=False) -> Union[None, Tuple[str, str, str, str, str, str, str]]:
        """
        Returns:  summary statistic regarding the F-distribution which contains the following parts of the distribution:
                (mean, median, mode, var, std, skewness, kurtosis). If the display parameter is True, the function returns None
                and prints out the summary of the distribution. 
        """
        if display:
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

    def keys(self) -> Dict[str, Union[float, int, str]]:
        """
        Summary statistic regarding the F-distribution which contains the following parts of the distribution:
        (mean, median, mode, var, std, skewness, kurtosis).

        Returns:
            Dict[str, Union[float, int, str]]: [description]
        """
        return {
            'main': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.main(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }
