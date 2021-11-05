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
            if not isinstance(x, (_np.ndarray, List)):
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
            if not isinstance(x, (_np.ndarray, List)):
                raise TypeError(
                    f'parameter x only accepts List types or numpy.ndarray')
            else:
                x = _np.array(x)
                return _gammainc(df/2, x/2)

        return _gammainc(df/2, randvar/2)

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

    def summary(self) -> Dict[str, Union[float, int, str]]:
        """
        Summary statistic regarding the ChiSquare-distribution which contains the following parts of the distribution:
        (mean, median, mode, var, std, skewness, kurtosis).

        Returns:
            Dict[str, Union[float, int, str]]
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }
