try:
    import numpy as _np
    from math import log as _log, log10 as _log10, pi as _pi, atan as _atan
    from typing import Union, Tuple, Dict, List
    from univariate._base import Infinite
except Exception as e:
    print(f"some modules are missing {e}")


class Cauchy(Infinite):
    """
    This class contains methods concerning the Cauchy Distribution [#]_ [#]_.

    .. math::
        \\text{Cauchy}(x;loc, scale) = \\frac{1}{\\pi \cdot scale \\big[ 1 + \\big( \\frac{x-loc}{scale} \\big)^2 \\big]}

    Args:

        loc(float): pertains to the loc parameter or median
        scale(float): pertains to  the scale parameter where scale > 0
        x(float): random variable

    References:
        .. [#] Wikipedia contributors. (2020, November 29). Cauchy distribution. https://en.wikipedia.org/w/index.php?title=Cauchy_distribution&oldid=991234690
        .. [#] Weisstein, Eric W. "Cauchy Distribution." From MathWorld--A Wolfram Web Resource. https://mathworld.wolfram.com/CauchyDistribution.html
    """

    def __init__(self, x: Union[float, int], loc: Union[float, int], scale: Union[float, int]):
        # if (type(x) and type(loc) and type(scale)) not in (int, float):
        #     raise TypeError('arguments must be of type int or float.')
        if scale < 0:
            raise ValueError(
                f'scale should be greater than 0. Entered value for scale:{scale}')
        self.scale = scale
        self.loc = loc
        self.x = x

    def pdf(self, x: Union[List[float], _np.ndarray] = None) -> Union[float, _np.ndarray]:
        """
        Args:

            x (List[float], numpy.ndarray): random variable or list of random variables

        Returns:
            either probability density evaluation for some point or plot of Cauchy distribution.
        """
        randvar = self.x
        loc = self.loc
        scale = self.scale

        if x is not None:
            if not isinstance(x, (_np.ndarray, List)):
                raise TypeError(f'parameter x only accepts List types or numpy.ndarray')
            else:
                x = _np.array(x)
                return 1/(_pi * scale * (1 + _np.power((x - loc) / scale, 2)))

        return 1/(_pi * scale * (1 + pow((randvar - loc) / scale, 2)))

    def cdf(self, x: Union[List[float], _np.ndarray] = None) -> Union[float, _np.ndarray]:
        """
        Args:

            x (List[float], numpy.ndarray): random variable or list of random variables

        Returns:
            either cumulative distribution evaluation for some point or plot of Cauchy distribution.
        """
        randvar = self.x
        loc = self.loc
        scale = self.scale

        def __generator(x, loc, scale): 
            return (1 / _pi) * _np.arctan((x - loc) / scale) + 1 / 2

        if x is not None:
            if not isinstance(x, (_np.ndarray, List)):
                raise TypeError(f'parameter x only accepts List types or numpy.ndarray')
            else:
                x = _np.array(x)
                return (1 / _pi) * _np.arctan((x - loc) / scale) + 1 / 2

        return (1 / _pi) * _atan((x - loc) / scale) + 1 / 2

    def mean(self) -> str:
        """
        Returns: Mean of the Cauchy distribution. Mean is Undefined.
        """
        return "undefined"

    def median(self) -> float:
        """
        Returns: Median of the Cauchy distribution.
        """
        return self.loc

    def mode(self) -> float:
        """
        Returns: Mode of the Cauchy distribution
        """
        return self.loc

    def var(self) -> str:
        """
        Returns: Variance of the Cauchy distribution.
        """
        return "undefined"

    def std(self) -> str:
        """
        Returns: Standard Deviation of the Cauchy Distribution.
        """
        return "undefined"

    def skewness(self) -> str:
        """
        Returns: Skewness of the Cauchy distribution.
        """
        return "undefined"

    def kurtosis(self) -> float:
        """
        Returns: Kurtosis of the Cauchy distribution
        """
        return _log(4 * _pi * self.scale)

    def entropy(self) -> float:
        """
        Returns: differential entropy of the Cauchy distribution

        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier.
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        return _log10(4*_pi*self.scale)

    def summary(self) -> Dict[str, Union[float, str]]:
        """
        Returns:
            Dictionary of Cauchy distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }
