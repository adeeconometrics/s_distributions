try:
    from scipy.special import beta as _beta, digamma as _digamma
    from scipy.integrate import quad as _quad
    import numpy as _np
    from math import sqrt as _sqrt, log as _log
    from typing import Union, Tuple, Dict, List
    from univariate._base import Infinite
except Exception as e:
    print(f"some modules are missing {e}")


class T(Infinite):
    """
    This class contains implementation of the Student's Distribution for calculating the
    probablity density function and cumulative distribution function. Additionally,
    a t-table __generator is also provided by p-value method. Note that the implementation
    of T(Student's) distribution is defined by beta-functions.

    Args:
        df(int): degrees of freedom.
        randvar(float): random variable.

    References:

    - Kruschke JK (2015). Doing Bayesian Data Analysis (2nd ed.). Academic Press. ISBN 9780124058880. OCLC 959632184.
    - Weisstein, Eric W. "Student's t-Distribution." From MathWorld--A Wolfram Web Resource. https://mathworld.wolfram.com/Studentst-Distribution.html

    """

    def __init__(self, df: int, randvar: float):
        if (type(df) is not int) or df < 0:
            raise TypeError(
                f'degrees of freedom(df) should be a whole number. Entered value for df: {df}')
        self.df = df
        self.randvar = randvar

    def pdf(self, x: Union[List[float], _np.ndarray] = None) -> Union[float, _np.ndarray]:
        """
        Args:

            x (List[float], numpy.ndarray): random variable or list of random variables

        Returns:
            either probability density evaluation for some point or plot of T distribution.
        """ 
        df = self.df
        randvar = self.randvar

        if x is not None:
            if not isinstance(x, (_np.ndarray, List)):
                raise TypeError(f'parameter x only accepts List types or numpy.ndarray')
            else:
                x = _np.array(x)
                return (1 / (_sqrt(df) * _beta(1 / 2, df / 2))) * _np.power((1 + _np.power(x, 2) / df), -(df + 1) / 2)

        return (1 / (_sqrt(df) * _beta(1 / 2, df / 2))) * pow((1 + pow(x, 2) / df), -(df + 1) / 2)

    def cdf(self, x: Union[List[float], _np.ndarray] = None) -> Union[float, List]:
        """
        Args:

            x (List[float], numpy.ndarray): random variable or list of random variables

        Returns:
            either cumulative distribution evaluation for some point or plot of T distribution.
        """ 
        df = self.df
        randvar = self.randvar
        # Test this for possible performance penalty. See if there is better way to do this.

        def __generator(x:float, df:int) -> float:
            def ___generator(x:float, df:int) -> float: 
                return (1 / (_sqrt(df) * _beta(1 / 2, df / 2))) * pow(1 + pow(x, 2) / df, -(df + 1) / 2)
            return _quad(___generator, -_np.inf, x, args=df)[0]

        if x is not None:
            if not isinstance(x, (_np.ndarray, List)):
                raise TypeError(f'parameter x only accepts List types or numpy.ndarray')
            else:
                return [__generator(i, df) for i in x]

        return __generator(randvar, df)

    def mean(self) -> Union[float, str]:
        """
        Returns:
            Mean of the T-distribution.

                0 for df > 1, otherwise undefined
        """
        df = self.df
        if df > 1:
            return 0.0
        return "undefined"

    def median(self) -> float:
        """
        Returns: Median of the T-distribution
        """
        return 0.0

    def mode(self) -> float:
        """
        Returns: Mode of the T-distribution
        """
        return 0.0

    def var(self) -> Union[float, str]:
        """
        Returns: Variance of the T-distribution
        """
        df = self.df
        if df > 2:
            return df / (df - 2)
        if df > 1 and df <= 2:
            return _np.inf
        return "undefined"

    def std(self) -> Union[float, str]:
        """
        Returns: Standard Deviation of the T-distribution
        """
        if self.var() == "undefined":
            return "undefined"
        return _sqrt(self.var())

    def skewness(self) -> Union[float, str]:
        """
        Returns: Skewness of the T-distribution
        """
        df = self.df
        if df > 3:
            return 0.0
        return "undefined"

    def kurtosis(self) -> Union[float, str]:
        """
        Returns: Kurtosis of the T-distribution
        """
        df = self.df
        if df > 4:
            return 6 / (df - 4)
        if df > 2 and df <= 4:
            return _np.inf
        return "undefined"

    def entropy(self) -> float:
        """
        Returns: differential entropy of T-distribution

        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier.
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        df = self.df
        return (df+1)/2 * (_digamma((df+1)/2)-_digamma(df/2)) + _log(_sqrt(df)*_beta(df/2, 1/2))

    def summary(self) -> Dict[str, Union[float, str]]:
        """
        Summary statistic regarding the T distribution which contains the following parts of the distribution:
        (mean, median, mode, var, std, skewness, kurtosis).

        Returns:
            Dict[str, Union[float, str]]
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }