try:
    import numpy as _np
    from math import sqrt as _sqrt, log as _log
    from typing import Union, Tuple, Dict, List
    from univariate._base import BoundedInterval
except Exception as e:
    print(f"some modules are missing {e}")


class Uniform(BoundedInterval):
    """
    This class contains methods concerning the Continuous Uniform Distribution [#]_.

    .. math::
        \\text{Uniform}(x;a,b) = \\frac{1}{b-a}

    Args:

        a(float): lower limit of the distribution 
        b(float): upper limit of the distribution where b > a

    Referene:
        .. [#] Weisstein, Eric W. "Uniform Distribution." From MathWorld--A Wolfram Web Resource. https://mathworld.wolfram.com/UniformDistribution.html
    """

    def __init__(self, a: float, b: float) -> None:
        self.a = a
        self.b = b

    def pdf(self, x: Union[List[float], _np.ndarray, float]) -> Union[float, _np.ndarray]:
        """
        Args:
            x (Union[List[float], numpy.ndarray, float]): random variable(s)

        Returns:
            Union[float, numpy.ndarray]: evauation of pdf at x
        """
        a = self.a
        b = self.b

        if isinstance(x, (_np.ndarray, List)):
            x0 = 1/(b-a)
            if isinstance(x, List):
                x = _np.array(x)
            return _np.piecewise(x, [_np.logical_and(a <= x, x <= b), _np.logical_or(a > x, x > b)], [lambda x: x0, lambda x: 0.0])

        return 1 / (b - a) if a <= x and x <= b else 0.0

    def cdf(self, x: Union[List[float], _np.ndarray, float]) -> Union[float, _np.ndarray]:
        """
        Args:
            x (Union[List[float], numpy.ndarray, float]): data point(s) of interest

        Returns:
            Union[float, numpy.ndarray]: evaluation of cdf at x
        """
        a = self.a
        b = self.b

        def __generator(a: float, b: float, x: float) -> float:
            if x < a:
                return 0.0
            if a <= x and x <= b:
                return (x - a) / (b - a)
            if x > b:
                return 1.0

        if isinstance(x, (_np.ndarray, List)):
            if isinstance(x, List):
                x = _np.array(x)
            # performance could be improved with np.piecewise
            return _np.vectorize(__generator)(a, b, x)

        return __generator(a, b, x)

    def mean(self) -> float:
        """
        Returns: Mean of the Uniform distribution.
        """
        return 1 / 2 * (self.a + self.b)

    def median(self) -> float:
        """
        Returns: Median of the Uniform distribution.
        """
        return 1 / 2 * (self.a + self.b)

    def mode(self) -> Tuple[int, int]:
        """
        Returns: Mode of the Uniform distribution.

        Note that the mode is any value in (a,b)
        """
        return (self.a, self.b)

    def var(self) -> float:
        """
        Returns: Variance of the Uniform distribution.
        """
        return 1 / 12 * pow(self.b - self.a, 2)

    def std(self) -> float:
        """
        Returns: Standard deviation of the Uniform distribution.
        """
        return _sqrt(1 / 12 * pow(self.b - self.a, 2))

    def skewness(self) -> float:
        """
        Returns: Skewness of the Uniform distribution.
        """
        return 0.0

    def kurtosis(self) -> float:
        """
        Returns: Kurtosis of the Uniform distribution.
        """
        return -6 / 5

    def entropy(self) -> float:
        """
        Returns: entropy of uniform Distirbution.

        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier.
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        return _log(self.b-self-a)

    def summary(self) -> Dict[str, Union[float, Tuple[int, int]]]:
        """
        Returns:
            Dictionary of Uniform distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }
