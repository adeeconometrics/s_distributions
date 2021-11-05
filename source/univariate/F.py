try:
    from scipy.special import beta as _beta, betainc as _betainc, gamma as _gamma, digamma as _digamma
    import numpy as _np
    from typing import Union, Tuple, Dict, List
    from math import sqrt as _sqrt, log as _log
    from univariate._base import SemiInfinite
except Exception as e:
    print(f"some modules are missing {e}")


class F(SemiInfinite):
    """
    This class contains methods concerning the F-distribution.

    Args:

        x(float | [0,infty)): random variable
        df1(int | x>0): first degrees of freedom
        df2(int | x>0): second degrees of freedom

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
        self.df2 = df2

    def pdf(self, x: Union[List[float], _np.ndarray] = None) -> Union[float, _np.ndarray]:
        """
        Args:

            x (List[float], numpy.ndarray): random variable or list of random variables

        Returns:
            either probability density evaluation for some point or plot of F-distribution.
        """

        def __generator(x:Union[float, _np.ndarray], df1:int, df2:int) -> Union[float, _np.ndarray]: 
            if type(x) is __np.ndarray:
                return (1 / _beta(df1 / 2, df2 / 2)) * pow(df1 / df2, df1 / 2) * \
                    _np.power(x, df1 / 2 - 1) * _np.power(1 + (df1 / df2) * x, -((df1 + df2) / 2))
            return (1 / _beta(df1 / 2, df2 / 2)) * pow(df1 / df2, df1 / 2) * \
                    pow(x, df1 / 2 - 1) * pow(1 + (df1 / df2) * x, -((df1 + df2) / 2))

        if x is not None:
            if not isinstance(x, (_np.ndarray, List)):
                raise TypeError(f'parameter x only accepts List types or numpy.ndarray')
            else:
                x = _np.array(x)
                return __generator(self.x, self.df1, self.df2)

        return __generator(self.randvar, self.df1, self.df2)

    def cdf(self,x: Union[List[float], _np.ndarray] = None) -> Union[float, _np.ndarray]:
        """
        Args:

            x (List[float], numpy.ndarray): random variable or list of random variables

        Returns:
            either cumulative distribution evaluation for some point or plot of F-distribution.
        """ 
        def __generator(x:Union[float, _np.ndarray], df1:int, df2:int) -> Union[float, _np.ndarray]: 
            return 1 - _betainc(df1/2, df2/2, x)

        if x is not None:
            if not isinstance(x, (_np.ndarray, List)):
                raise TypeError(f'parameter x only accepts List types or numpy.ndarray')
            else:
                x = _np.array(x)
                return __generator(x, self.df1, self.df2)

        k = self.df2/(self.df2 + self.df1*self.x)
        return __generator(k, self.df1, self.df2)

    def mean(self) -> Union[float, int, str]:
        """
        Returns: Mean of the F-distribution.
        """
        if self.df2 > 2:
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
            return (2 * pow(df2, 2) * (df1 + df2 - 2)) / (df1 * (pow(df2 - 2, 2) *
                                                       (df2 - 4)))
        return "undefined"

    def std(self) -> Union[float, str]:
        """
        Returns: Standard deviation of the F-distribution.
        """
        df1 = self.df1
        df2 = self.df2
        if df2 > 4:
            return _sqrt((2 * pow(df2, 2) * (df1 + df2 - 2))/(df1 * (pow(df2 - 2, 2) * (df2 - 4))))
        return 'undefined'

    def skewness(self) -> Union[float, str]:
        """
        Returns: Skewness of the F-distribution.
        """
        df1=self.df1
        df2=self.df2
        if df2 > 6:
            return ((2 * df1 + df2 - 2) * _sqrt(8 * (df2 - 4))) / ((df2 - 6) * _sqrt(df1 * (df1 + df2 - 2)))
        return "undefined"

    def entropy(self) -> Union[float, int]:
        """
        Returns: differential entropy of F-distribution.

        Reference: Lazo, A.V.; Rathie, P. (1978). "On the entropy of continuous probability distributions". IEEE Transactions on Information Theory
        """
        df1=self.df1
        df2=self.df2
        return _log(_gamma(df1/2)) + _log(_gamma(df2/2)) -\
            _log(_gamma((df1+df2)/2)) + (1-df1/2)*_digamma(1+df1/2) -\
            (1-df2/2) * _digamma(1+df2/2) + (df1+df2) /\
             2*_digamma((df1+df2)/2) + _log(df1/df2)

    def keys(self) -> Dict[str, Union[float, int, str]]:
        """
        Summary statistic regarding the F-distribution which contains the following parts of the distribution:
        (mean, median, mode, var, std, skewness, kurtosis).

        Returns:
            Dict[str, Union[float, int, str]]
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }
