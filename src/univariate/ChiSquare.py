try:
    from scipy.special import gammainc as _gammainc, gamma as _gamma, digamma as _digamma
    import numpy as _np
    from math import sqrt as _sqrt, log as _log
    from typing import Union, Tuple, Dict, List
    from univariate._base import SemiInfinite
except Exception as e:
    print(f"some modules are missing {e}")


class ChiSquare(SemiInfinite):
    """
    This class contains methods concerning the Chi-square distribution.

    Args:

        x(float): random variable.
        df(int): degrees of freedom.

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
        - summary for printing the summary statistics of the distribution.
        - keys for returning a dictionary of summary statistics.

    References:
    - Weisstein, Eric W. "Chi-Squared Distribution." From MathWorld--A Wolfram Web Resource.
    https://mathworld.wolfram.com/Chi-SquaredDistribution.html
    - Wikipedia contributors. (2020, December 13). Chi-square distribution. In Wikipedia, The Free Encyclopedia.
    Retrieved 04:37, December 23, 2020, from https://en.wikipedia.org/w/index.php?title=Chi-square_distribution&oldid=994056539
    """

    def __init__(self, df: int, randvar: Union[float, int] = 0.0):
        if type(df) is not int:
            raise TypeError('degrees of freedom(df) should be a whole number.')
        if df < 0:
            raise ValueError(
                f'Entered value for df: {df}, it should be a positive integer.')

        self.randvar = randvar
        self.df = df

    def pdf(self, x: Union[List[float], _np.ndarray] = None) -> Union[float, _np.ndarray]:
        """
        Args:

            x (List[float], numpy.ndarray): random variable or list of random variables

        Returns:
            either probability density evaluation for some point or plot of Chi square-distribution.
        """
        randvar = self.randvar
        df = self.df

        if x is not None:
            if not (isinstance(x, _np.ndarray)) and issubclass(x, List):
                raise TypeError(
                    f'parameter x only accepts List types or numpy.ndarray')
            else:
                x = _np.array(x)
                return (1 / (_np.pow(2, (df / 2) - 1) * _gamma(df / 2))) * _np.pow(x, df - 1) * _np.exp(-_np.pow(x, 2) / 2)

        return (1 / (pow(2, (df / 2) - 1) * _gamma(df / 2))) * pow(x, df - 1) * _np.exp(-pow(x, 2) / 2)

    def cdf(self, x: Union[List[float], _np.ndarray] = None) -> Union[float, _np.ndarray]:
        """
        Args:

            x (List[float], numpy.ndarray): random variable or list of random variables

        Returns:
            either cumulative distribution evaluation for some point or plot of Chi square-distribution.
        """
        randvar = self.randvar
        df = self.df

        if x is not None:
            if not (isinstance(x, _np.ndarray)) and issubclass(x, List):
                raise TypeError(
                    f'parameter x only accepts List types or numpy.ndarray')
            else:
                x = _np.array(x)
                return _gammainc(df/2, x/2)

        return _gammaince(df/2, randvar/2)

    def p_val(self, x_lower=-_np.inf, x_upper=None) -> float:
        """
        Args:

            x_lower(float): defaults to -_np.inf. Defines the lower value of the distribution. Optional.
            x_upper(float | x_upper>x_lower): defaults to None. If not defined defaults to random variable x. Optional.
            args(list of float): pvalues of each elements from the list

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Chi square distribution evaluated at some random variable.
        """
        def __cdf(x, df): return _gammainc(df / 2, x / 2)
        if x_upper is not None:
            if x_lower > x_upper:
                raise Exception('x_lower should be less than x_upper.')
            return __cdf(x_upper, self.df) - __cdf(x_lower, self.df)
        return __cdf(self.randvar, self.df)

    def mean(self) -> Union[float, int]:
        """
        Returns: Mean of the Chi-square distribution.
        """
        return self.df

    def median(self) -> Union[float, int]:
        """
        Returns: Median of the Chi-square distribution.
        """
        return self.df * pow(1 - 2 / (9 * self.df), 3)

    def var(self) -> Union[float, int]:
        """
        Returns: Variance of the Chi-square distribution.
        """
        return 2 * self.df

    def std(self) -> float:
        """
        Returns: Standard deviation of the Chi-square distribution.
        """
        return _sqrt(2 * self.df)

    def skewness(self) -> float:
        """
        Returns: Skewness of the Chi-square distribution.
        """
        return _sqrt(8 / self.df)

    def kurtosis(self) -> float:
        """
        Returns: Kurtosis of the Chi-square distribution.
        """
        return 12 / self.df

    def entropy(self) -> float:
        """
        Returns: differential entropy of Chi-square distribution.

        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier.
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        df = self.df
        return df/2 + _log(2*_gamma(df/2)) + (1-df/2)*_digamma(df/2)

    def summary(self, display=False) -> Union[None, Tuple[str, str, str, str, str, str, str]]:
        """
        Returns:  summary statistic regarding the ChiSquare-distribution which contains the following parts of the distribution:
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

    def keys(self) -> Dict[str, Union[float, int, str]]:
        """
        Summary statistic regarding the ChiSquare-distribution which contains the following parts of the distribution:
        (mean, median, mode, var, std, skewness, kurtosis).

        Returns:
            Dict[str, Union[float, int, str]]: [description]
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }
