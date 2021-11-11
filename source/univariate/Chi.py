try:
    from scipy.special import gammainc as _gammainc, gamma as _gamma, digamma as _digamma
    import numpy as _np
    from math import sqrt as _sqrt, log as _log
    from typing import Union, Dict, List
    from univariate._base import SemiInfinite
except Exception as e:
    print(f"some modules are missing {e}")


class Chi(SemiInfinite):
    """
    This class contains methods concerning the Chi distribution [#]_ [#]_.

    .. math:: 
        \\text{Chi}(x;df) = {\\frac{1}{2^{(df/2)-1}\\Gamma(df/2)} \\cdot x^{df-1} e^{-x^2/2}}

    Args:

        df(int): degrees of freedom where df > 0
        x(float): random variable

    References:
        .. [#] Weisstein, Eric W. "Chi Distribution." From MathWorld--A Wolfram Web Resource. https://mathworld.wolfram.com/ChiDistribution.html
        .. [#] Wikipedia contributors. (2020, October 16). Chi distribution. https://en.wikipedia.org/w/index.php?title=Chi_distribution&oldid=983750392
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
            if not isinstance(x, (_np.ndarray, List)):
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
            if not isinstance(x, (_np.ndarray, List)):
                raise TypeError(f'parameter x only accepts List types or numpy.ndarray')
            else:
                x = _np.array(x)
                return _gammainc(df/2, _np.power(x, 2)/2)
        return _gammainc(df/2, pow(randvar, 2)/2)

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
        return _log(_gamma(df/2)/_sqrt(2)) - (df-1)/2*_digamma(df/2) + df/2


    def summary(self) -> Dict[str, Union[float, int, str]]:
        """
        Returns:
            Dictionary of Chi distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }
