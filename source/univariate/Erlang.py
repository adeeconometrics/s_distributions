try:
    from scipy.special import gamma as _gamma, gammainc as _gammainc, digamma as _digamma
    import numpy as _np
    from typing import Union, Tuple, Dict, List
    from math import sqrt as _sqrt, log as _log, factorial as _factorial
    from univariate._base import SemiInfinite
except Exception as e:
    print(f"some modules are missing {e}")


class Erlang(SemiInfinite):
    """
    This class contains methods concerning Erlang Distirbution [#]_ [#]_.

    .. math:: 
        \\text{Erlang}(x; k, \\lambda) = \\frac{\\lambda^{k} x^{k-1} e^{\\lambda x}}{(k-1)!}

    Args:

        shape(int): shape parameter (:math:`k`) where shape > 0
        rate(float): rate parameter (:math:`\\lambda`) where rate >= 0
        randvar(float): random variable where x >= 0

    Reference:
        .. [#] Wikipedia contributors. (2021, January 6). Erlang distribution. https://en.wikipedia.org/w/index.php?title=Erlang_distribution&oldid=998655107
        .. [#] Weisstein, Eric W. "Erlang Distribution." From MathWorld--A Wolfram Web Resource. https://mathworld.wolfram.com/ErlangDistribution.html
    """

    def __init__(self, shape: int, rate: float, randvar: float):
        if randvar < 0:
            raise ValueError(
                f'random variable should only be in between 0 and 1. Entered value: {randvar}')
        if isinstance(shape, int) == False and shape > 0:
            raise TypeError(
                'shape parameter should be an integer greater than 0.')
        if rate < 0:
            raise ValueError(
                f'beta parameter(rate) should be a positive number. Entered value: {rate}')

        self.shape = shape
        self.rate = rate
        self.randvar = randvar

    def pdf(self,x: Union[List[float], _np.ndarray] = None) -> Union[float, _np.ndarray]:
        """
        Args:

            x (List[float], numpy.ndarray): random variable or list of random variables

        Returns:
            either probability density evaluation for some point or plot of Erlang distribution.
        """ 
        shape = self.shape
        rate = self.rate
        randvar = self.randvar

        if x is not None:
            if not isinstance(x, (_np.ndarray, List)):
                raise TypeError(f'parameter x only accepts List types or numpy.ndarray')
            else:
                x = _np.array(x)
                return pow(rate, shape)* _np.power(x, shape-1)*_np.exp(-rate*x) / _factorial(shape-1)

        return pow(rate, shape)*pow(randvar, shape-1)*_exp(-rate*randvar)/_factorial(shape-1)

    def cdf(self, x: Union[List[float], _np.ndarray] = None) -> Union[float, _np.ndarray]:
        """
        Args:

            x (List[float], numpy.ndarray): random variable or list of random variables

        Returns:
            either cumulative distribution evaluation for some point or plot of Erlang distribution.
        """
        shape = self.shape
        rate = self.rate
        randvar = self.randvar

        def __generator(shape, rate, x): 
            return _gammainc(shape, rate*x)/_factorial(shape-1)

        if x is not None:
            if not isinstance(x, (_np.ndarray, List)):
                raise TypeError(f'parameter x only accepts List types or numpy.ndarray')
            else:
                x = _np.array(x)
                return __generator(shape, rate, x)

        return __generator(shape, rate, randvar)

    def mean(self) -> float:
        """
        Returns: Mean of the Erlang distribution.
        """
        return self.shape/self.rate

    def median(self) -> str:
        """
        Returns: Median of the Erlang distribution.
        """
        return "no simple closed form"

    def mode(self) -> Union[float, str]:
        """
        Returns: Mode of the Erlang distribution.
        """
        return (1/self.rate)*(self.shape-1)

    def var(self) -> float:
        """
        Returns: Variance of the Erlang distribution.
        """
        return self.shape/pow(self.rate, 2)

    def std(self) -> float:
        """
        Returns: Standard deviation of the Eerlang distribution.
        """
        return _sqrt(self.var())

    def skewness(self) -> float:
        """
        Returns: Skewness of the Erlang distribution.
        """
        return 2/_sqrt(self.shape)

    def kurtosis(self) -> float:
        """
        Returns: Kurtosis of the Erlang distribution.
        """
        return 6/self.shape

    def entropy(self) -> float:
        """
        Returns: differential entropy of the Erlang distribution.

        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier.
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        k = self.shape
        _lambda = self.rate
        return (1-k)*_digamma(k)+_log(_gamma(k)/_lambda)+k

    def summary(self) -> Dict[str, Union[float, int, str]]:
        """
        Returns:
            Dictionary of Erlang distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }

