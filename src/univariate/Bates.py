try:
    from scipy.special import binom as _binom
    import numpy as np
    from typing import Union, Tuple, Dict, List
    from math import sqrt as _sqrt, factorial as _factorial
    from univariate._base import BoundedInterval
except Exception as e:
    print(f"some modules are missing {e}")


class Bates(BoundedInterval):
    """
    This class contains methods concerning Bates Distirbution. Also referred to as the regular mean distribution.

    Note that the Bates distribution is a probability distribution of the mean of a number of statistically indipendent uniformly
    distirbuted random variables on the unit interval. This is often confused with the Irwin-Hall distribution which is
    the distribution of the sum (not the mean) of n independent random variables. The two distributions are simply versions of
    each other as they only differ in scale.
    Args:

        a(float): lower bound
        b(float |b>a): upper bound
        n(int | x>=1)
        randvar(float | [a,b]): random variable

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

    Reference:
    - Wikipedia contributors. (2021, January 8). Bates distribution. In Wikipedia, The Free Encyclopedia.
    Retrieved 08:27, January 8, 2021, from https://en.wikipedia.org/w/index.php?title=Bates_distribution&oldid=999042206
    """

    def __init__(self, a: float, b: float, n: int, randvar: float):
        if randvar < 0 or randvar > 1:
            raise ValueError(
                f'random variable should only be in between 0 and 1. Entered value: {randvar}')
        if a > b:
            raise ValueError(
                'lower bound (a) should not be greater than upper bound (b).')
        if type(n) is not int:
            raise TypeError('parameter n should be an integer type.')

        self.a = a
        self.b = b
        self.n = n
        self.randvar = randvar

    def pdf(self, x: Union[List[float], _np.ndarray] = None) -> Union[float, _np.ndarray]:
        """
        Args:

            x (List[float], numpy.ndarray): random variable or list of random variables

        Returns:
            either probability density evaluation for some point or plot of Bates distribution.
        """
        # def __generator(a, b, n, x):
        #     if a < x or x < b:
        #         return sum(pow(-1, i)*_binom(n, i)*pow(((x-a)/(b-a) - i/n), n-1)*np.sign((x-a)/(b-1)-i/n) for i in range(1, n+1))
        #     return 0

        # if plot:
        #     x = np.linspace(0, 1, int(threshold))
        #     y = np.array([__generator(self.a, self.b, self.n, i) for i in x])
        #     return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        # return __generator(self.a, self.b, self.n, self.randvar)

    def cdf(self, x: Union[List[float], _np.ndarray] = None) -> Union[float, _np.ndarray]:
        """
        Args:

            x (List[float], numpy.ndarray): random variable or list of random variables

        Returns:
            either cumulative distribution evaluation for some point or plot of Bates distribution.
        """
        return "currently unsupported"

    def pvalue(self, x_lower=0, x_upper=None) -> Optional[float]:
        """
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Bates distribution evaluated at some random variable.
        """

        return "currently unsupported"

    def mean(self) -> float:
        """
        Returns: Mean of the Bates distribution.
        """
        return 0.5*(self.a+self.b)

    def var(self) -> float:
        """
        Returns: Variance of the Bates distribution.
        """
        return 1/(12*self.n)*pow(self.b-self.a, 2)

    def std(self) -> float:
        """
        Returns: Standard devtiation of the Bates distribution
        """
        return _sqrt(1/(12*self.n)*pow(self.b-self.a, 2))

    def skewness(self) -> float:
        """
        Returns: Skewness of the Bates distribution.
        """
        return -6/(5*self.n)

    def kurtosis(self) -> float:
        """
        Returns: Kurtosis of the Bates distribution.
        """
        return 0.0

    def summary(self, display=False) -> Union[None, Tuple[str, str, str, str, str, str, str]]:
        """
        Returns:  summary statistic regarding the Bates distribution which contains the following parts of the distribution:
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
        Summary statistic regarding the Bates distribution which contains the following parts of the distribution:
        (mean, median, mode, var, std, skewness, kurtosis).

        Returns:
            Dict[str, Union[float, Tuple[float]]]: [description]
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }
