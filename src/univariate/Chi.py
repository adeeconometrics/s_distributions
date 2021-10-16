try:
    from scipy.special import gammainc as _gammainc, gamma as _gamma
    import numpy as __np
    from math import sqrt as _sqrt, log as _log
    from typing import Union, Tuple, Dict, List
    from _base import SemiInfinite
except Exception as e:
    print(f"some modules are missing {e}")


class Chi(SemiInfinite):
    """
    This class contains methods concerning the Chi distribution.

    Args:

        x(float): random variable.
        df(int | x>0): degrees of freedom.

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
    - Weisstein, Eric W. "Chi Distribution." From MathWorld--A Wolfram Web Resource.
    https://mathworld.wolfram.com/ChiDistribution.html
    - Wikipedia contributors. (2020, October 16). Chi distribution. In Wikipedia, The Free Encyclopedia.
    Retrieved 10:35, January 2, 2021, from https://en.wikipedia.org/w/index.php?title=Chi_distribution&oldid=983750392
    """

    def __init__(self, df:int, randvar:float):
        if type(df) is not int:
            raise TypeError('degrees of freedom(df) should be a whole number.')

        if df <= 0:
            raise ValueError(
                f'Entered value for df: {df}, it should be a positive integer.')

        self.randvar = randvar
        self.df = df

    def pdf(self, x: Union[List[float], _np.ndarray] = None) -> Union[float, _np.ndarray]:
        """
        Args:

            x (List[float], numpy.ndarray): random variable or list of random variables

        Returns:
            either probability density evaluation for some point or plot of Chi-distribution.

        """
        df = self.df
        randvar = self.randvar
        
        if x is not None:
            if not (isinstance(x, _np.ndarray)) and issubclass(x, List):
                raise TypeError(f'parameter x only accepts List types or numpy.ndarray')
            else:
                x = _np.array(x)
                return (1 / (pow(2, (df / 2) - 1) * _gamma(df / 2))) * _np.power(x, df - 1) * _np.exp(_np.power(-x, 2) / 2)

        return (1 / (pow(2, (df / 2) - 1) * _gamma(df / 2))) * pow(randvar, df - 1) * _np.exp(pow(-randvar, 2) / 2)

    def cdf(self, x: Union[List[float], _np.ndarray] = None) -> Union[float, _np.ndarray]:
        """
        Args:

            x (List[float], numpy.ndarray): random variable or list of random variables

        Returns:
            either cumulative distribution evaluation for some point or plot of Chi-distribution.
        """
        randvar = self.randvar
        df = self.df

        if x is not None:
            if not (isinstance(x, _np.ndarray)) and issubclass(x, List):
                raise TypeError(f'parameter x only accepts List types or numpy.ndarray')
            else:
                x = _np.array(x)
                return _gammainc(df/2, _np.power(x, 2)/2)
        return _gammainc(df/2, pow(randvar, 2)/2)

    def p_val(self, x_lower=-_np.inf, x_upper=None) -> Optional[float]:
        """
        Args:

            x_lower(float): defaults to -_np.inf. Defines the lower value of the distribution. Optional.
            x_upper(float | x_upper>x_lower): defaults to None. If not defined defaults to random variable x. Optional.
            args(list of float): pvalues of each elements from the list

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Chi distribution evaluated at some random variable.
        """
        def __cdf(x, df): return _gammainc(df/2, pow(x, 2)/2)
        if x_upper != None:
            if x_lower > x_upper:
                raise Exception('x_lower should be less than x_upper.')
            return __cdf(x_upper, self.df) - __cdf(x_lower, self.df)
        return __cdf(self.randvar, self.df)

    def mean(self) -> float:
        """
        Returns: Mean of the Chi distribution.
        """
        return _sqrt(2)*_gamma((self.df+1)/2)/_gamma(self.df/2)

    def median(self) -> Union[float, int]:
        """
        Returns: Median of the Chi distribution.
        """
        return pow(self.df*(1-(2/(1*self.df))), 3/2)

    def mode(self) -> Union[float, str]:
        """
        Returns: Mode of the Chi distribution.
        """
        if self.df >= 1:
            return _sqrt(self.df-1)
        return "undefined"

    def var(self) -> Union[float, int]:
        """
        Returns: Variance of the Chi distribution.
        """
        return pow(self.df-self.mean(), 2)

    def std(self) -> Union[float, int]:
        """
        Returns: Standard deviation of the Chi distribution.
        """
        return self.df-self.mean()

    def skewness(self) -> Union[float, int]:
        """
        Returns: Skewness of the Chi distribution.
        """
        mean = self.mean()
        std = self.df - mean
        return (mean - 2*pow(std, 2))/pow(std, 3)

    def kurtosis(self) -> float:
        """
        Returns: Kurtosis of the Chi distribution.
        """
        mean = self.mean()
        var = pow(self.df-mean, 2)
        std = self.df - mean
        sk = (mean - 2*pow(std, 2))/pow(std, 3)

        return 2*(1-mean*_sqrt(var)*sk-var)/var

    def entropy(self) -> float:
        """
        Returns: differential entropy of Chi distribution.

        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier.
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        df = self.df
        return _log(_gamma(df/2)/_sqrt(2)) - (df-1)/2*digamma(df/2) + df/2

    def summary(self, display=False) -> Union[None, Tuple[str, str, str, str, str, str, str]]:
        """
        Returns:  summary statistic regarding the Chi-distribution which contains the following parts of the distribution:
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
        Summary statistic regarding the Chi-distribution which contains the following parts of the distribution:
        (mean, median, mode, var, std, skewness, kurtosis).

        Returns:
            Dict[str, Union[float, int]]: [description]
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }
