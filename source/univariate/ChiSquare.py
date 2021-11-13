try:
    from scipy.special import gammainc as _gammainc, gamma as _gamma, digamma as _digamma
    import numpy as _np
    from math import sqrt as _sqrt, log as _log
    from typing import Union, Dict, List
    from univariate._base import SemiInfinite
except Exception as e:
    print(f"some modules are missing {e}")


class ChiSquare(SemiInfinite):
    """
    This class contains methods concerning the Chi-square distribution [#]_ [#]_.

    .. math:: 
        \\text{ChiSquare}(x;k) = {\\frac{1}{2^{k/2}\\Gamma(k/2)}\\ x^{k/2-1}e^{-x/2}}

    Args:

        df(int): degrees of freedom (:math:`k`) where df > 0
        x(float): random variable.

    References:
        .. [#] Weisstein, Eric W. "Chi-Squared Distribution." From MathWorld--A Wolfram Web Resource. https://mathworld.wolfram.com/Chi-SquaredDistribution.html
        .. [#] Wikipedia contributors. (2020, December 13). Chi-square distribution. https://en.wikipedia.org/w/index.php?title=Chi-square_distribution&oldid=994056539
    """

    def __init__(self, df: int):
        if type(df) is not int:
            raise TypeError('degrees of freedom(df) should be a whole number.')
        if df < 0:
            raise ValueError('df should be a positive integer.')

        self.df = df

    def pdf(self, x: Union[List[float], _np.ndarray, float]) -> Union[float, _np.ndarray]:
        """
        Args:
            x (Union[List[float], numpy.ndarray, float]): random variable(s)

        Raises:
            ValueError: when there exist a vaue less than 0

        Returns:
            Union[float, numpy.ndarray]: evaluation of pdf at x
        """
        df = self.df

        if isinstance(x, (_np.ndarray, List)):
            x = _np.array(x)
            if _np.any(x < 0):
                raise ValueError(
                    'random variables are only valid for positive real numbers')
            return (1 / (_np.power(2, (df / 2) - 1) * _gamma(df / 2))) * _np.power(x, df - 1) * _np.exp(-_np.power(x, 2) / 2)

        if x < 0:
            raise ValueError(
                'random variable are only valid for positive real numbers')
        return (1 / (pow(2, (df / 2) - 1) * _gamma(df / 2))) * pow(x, df - 1) * _np.exp(-pow(x, 2) / 2)

    def cdf(self, x: Union[List[float], _np.ndarray, float]) -> Union[float, _np.ndarray]:
        """
        Args:
            x (Union[List[float], numpy.ndarray, float]): data point(s) of interest

        Raises:
            ValueError: when there exist a value of x less than 0

        Returns:
            Union[float, numpy.ndarray]: evaluation of cdf at x
        """
        df = self.df

        if isinstance(x, (_np.ndarray, List)):
            x = _np.array(x)
            if _np.any(x < 0):
                raise ValueError(
                    'data point(s) are only valid for positive real numbers')
            return _gammainc(df/2, x/2)

        if x < 0:
            raise ValueError(
                'data point(s) are only valid for positive real numbers')
        return _gammainc(df/2, x/2)

    def mean(self) -> float:
        """
        Returns: Mean of the Chi-square distribution.
        """
        return self.df

    def median(self) -> float:
        """
        Returns: Median of the Chi-square distribution.
        """
        return self.df * pow(1 - 2 / (9 * self.df), 3)

    def var(self) -> float:
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

    def summary(self) -> Dict[str, Union[float, str]]:
        """
        Returns:
            Dictionary of Chi-Square distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }
