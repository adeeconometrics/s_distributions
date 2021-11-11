try:
    import numpy as _np
    from math import sqrt as _sqrt, log as _log
    from typing import Union, Tuple, Dict, List
    from univariate._base import SemiInfinite
except ValueError as e:
    print(f"some modules are missing {e}")


class Pareto(SemiInfinite):
    """
    This class contains methods concerning the Pareto Distribution Type 1 [#]_ [#]_.

    .. math:: \\text{Pareto}(x;x_m, a) = \\frac{a x_m^a}{x^{a+1}}

    Args:

        scale(float): scale parameter (:math:`x_m`) where scale > 0
        shape(float): shape parameter (:math:`a`) where shape > 0
        x(float): random variable where shape <= x

    References:
        .. [#] Barry C. Arnold (1983). Pareto Distributions. International Co-operative Publishing House. ISBN 978-0-89974-012-6.
        .. [#] Wikipedia contributors. (2020, December 1). Pareto distribution. https://en.wikipedia.org/w/index.php?title=Pareto_distribution&oldid=991727349
    """

    def __init__(self, shape: float, scale: float, x: float):
        if scale < 0:
            raise ValueError(
                f'scale should be greater than 0. Entered value for scale:{scale}')
        if shape < 0:
            raise ValueError(
                f'shape should be greater than 0. Entered value for shape:{shape}')
        if x > shape:
            raise ValueError(
                f'random variable x should be greater than or equal to shape. Entered value for x:{x}')

        self.shape = shape
        self.scale = scale
        self.x = x

    def pdf(self, x: Union[List[float], _np.ndarray] = None) -> Union[float, _np.ndarray]:
        """
        Args:

            x (List[float], numpy.ndarray): random variable or list of random variables

        Returns:
            either probability density evaluation for some point or plot of Pareto distribution.
        """
        x_m = self.scale
        alpha = self.shape
        randvar = self.x

        def __generator(x: float, x_m: float, alpha: float) -> float:
            if x >= x_m:
                return (alpha * pow(x_m, alpha)) / pow(x, alpha + 1)
            return 0.0

        if x is not None:
            if not isinstance(x, (_np.ndarray, List)):
                raise TypeError(
                    f'parameter x only accepts List types or numpy.ndarray')
            else:
                return [__generator(i, x_m, alpha) for i in x]
        return __generator(randvar, x_m, alpha)

    def cdf(self, x: Union[List[float], _np.ndarray] = None) -> Union[float, _np.ndarray]:
        """
        Args:

            x (List[float], numpy.ndarray): random variable or list of random variables

        Returns:
            either cumulative distribution evaluation for some point or plot of Pareto distribution.
        """
        x_m = self.scale
        alpha = self.shape
        randvar = self.x

        def __generator(x: float, x_m: float, alpha: float) -> float:
            if x >= x_m:
                return 1 - pow(x_m / x, alpha)
            return 0.0

        if x is not None:
            if not isinstance(x, (_np.ndarray, List)):
                raise TypeError(
                    f'parameter x only accepts List types or numpy.ndarray')
            else:
                return [__generator(i, x_m, alpha) for i in x]
        return __generator(randvar, x_m, alpha)

    def mean(self) -> float:
        """
        Returns: Mean of the Pareto distribution.
        """
        a = self.shape
        x_m = self.scale

        if a <= 1:
            return _np.inf
        return (a * x_m) / (a - 1)

    def median(self) -> float:
        """
        Returns: Median of the Pareto distribution.
        """
        a = self.shape
        x_m = self.scale
        return x_m * pow(2, 1 / a)

    def mode(self) -> float:
        """
        Returns: Mode of the Pareto distribution.
        """
        return self.scale

    def var(self) -> float:
        """
        Returns: Variance of the Pareto distribution.
        """
        a = self.shape
        x_m = self.scale
        if a <= 2:
            return _np.inf
        return (pow(x_m, 2) * a) / (pow(a - 1, 2) * (a - 2))

    def std(self) -> float:
        """
        Returns: Variance of the Pareto distribution
        """
        return _sqrt(self.var())

    def skewness(self) -> Union[float, str]:
        """
        Returns: Skewness of the Pareto distribution.
        """
        a = self.shape
        x_m = self.scale
        if a > 3:
            scale = (2 * (1 + a)) / (a - 3)
            return scale * _sqrt((a - 2) / a)
        return "undefined"

    def kurtosis(self) -> Union[float, str]:
        """
        Returns: Kurtosis of the Pareto distribution.
        """
        a = self.shape
        x_m = self.scale
        if a > 4:
            return (6 * (a**3 + a**2 - 6 * a - 2)) / (a * (a - 3) * (a - 4))
        return "undefined"

    def entropy(self) -> float:
        """
        Returns: differential entropy of the Pareto distribution.

        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier.
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        a = self.shape
        x_m = self.scale
        return _log(x_m/a)+1+(1/a)

    def summary(self) -> Dict[str, Union[float, str]]:
        """
        Returns:
            Dictionary of Pareto distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }
