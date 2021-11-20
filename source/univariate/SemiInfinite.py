try:
    import numpy as np
    from numpy import euler_gamma as euler
    import scipy.special as ss
    from typing import Union, Tuple, Dict, List
    import math as m
    from univariate._base import Base
except Exception as e:
    print(f"some modules are missing {e}")


class SemiInfinite(Base):
    """
    Description:
        Base class for probability tags.
    """

    def __init__(self):
        if type(self) is SemiInfinite:
            raise TypeError('base class cannot be instantiated.')


class Weibull(SemiInfinite):
    """
    This class contains methods concerning Weibull Distirbution [#]_.

    .. math::
        \\text{Weibull}(x;\\lambda, k)  = \\frac{k}{\\lambda} \\Big( \\frac{x}{\\lambda}\\Big)^{k-1} \\exp(-(x/\\lambda)^k)

    Args:

        shape(float): shape parameter (:math:`\\lambda`) where shape >= 0
        scale(float): scale parameter (:math:`k`) where scale >= 0
        randvar(float): random variable where x >= 0

    Reference:
        .. [#] Wikipedia contributors. (2020, December 13). Weibull distribution. https://en.wikipedia.org/w/index.php?title=Weibull_distribution&oldid=993879185
    """

    def __init__(self, shape: float, scale: float):
        if shape < 0 or scale < 0:
            raise ValueError('all parameters should be a positive number.')
        self.scale = scale
        self.shape = shape

    def pdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]:
        """
        Args:

            x (List[float], numpy.ndarray): random variable or list of random variables

        Returns:
            either probability density evaluation for some point or plot of Weibull distribution.
        """
        scale = self.scale
        shape = self.shape

        if isinstance(x, (np.ndarray, List)):
            if not type(x) is np.ndarray:
                x = np.array(x)

            def f1(x): return np.power(shape/scale*x/scale, shape-1) * \
                np.exp(-np.power(x/scale, shape))
            return np.piecewise(x, [x < 0, x >= 0], [0.0, f1])

        return pow((shape/scale)*(x/scale), shape-1)*m.exp(-pow(x/scale, shape)) if x >= 0 else 0.0

    def cdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]:
        """
        Args:

            x (List[float], numpy.ndarray): random variable or list of random variables

        Returns:
            either cumulative distribution evaluation for some point or plot of Weibull distribution.
        """
        scale = self.scale
        shape = self.shape

        if isinstance(x, (np.ndarray, List)):
            if not type(x) is np.ndarray:
                x = np.array(x)

            def f1(x): return 1 - np.exp(-np.power(x/scale, shape))

            return np.piecewise(x, [x >= 0, x < 0], [f1, 0.0])

        return 1-m.exp(-pow(x/scale, shape)) if x >= 0 else 0.0

    def mean(self) -> float:
        """
        Returns: Mean of the Weibull distribution.
        """
        return self.scale*ss.gamma(1+(1/self.shape))

    def median(self) -> float:
        """
        Returns: Median of the Weibull distribution.
        """
        return self.scale*pow(m.log(2), 1/self.shape)

    def mode(self) -> float:
        """
        Returns: Mode of the Weibull distribution.
        """
        if self.shape > 1:
            return self.scale*pow((self.shape-1)/self.shape, 1/self.shape)
        return 0

    def var(self) -> float:
        """
        Returns: Variance of the Weibull distribution.
        """
        return pow(self.scale, 2) * pow(ss.gamma(1+2/self.shape) - ss.gamma(1+1/self.shape), 2)

    def std(self) -> float:
        """
        Returns: Standard deviation of the Weilbull distribution
        """
        return m.sqrt(pow(self.scale, 2) * pow(ss.gamma(1+2/self.shape) - ss.gamma(1+1/self.shape), 2))

    def entropy(self) -> float:
        """
        Returns: differential entropy of the Weilbull distribution.

        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier.
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        return (self.scale+1) * euler/self.scale + m.log(self.shape/self.scale) + 1

    def summary(self) -> Dict[str, Union[float, int, str]]:
        """
        Returns:
            Dictionary of Weibull distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }


class WeibullInverse(SemiInfinite):
    """
    This class contains methods concerning inverse Weilbull or the Fréchet Distirbution [#]_.

    .. math::
        \\text{WeibullInverse}(x;a,s,m) = \\frac{a}{s} \\Big(\\frac{x-m}{s} \\Big) ^{-1-a} \\exp{\\Big(-\\frac{x-m}{s} \\Big)^{-a}}

    Args:

        shape(float): shape parameter (:math:`a`) where shape >= 0
        scale(float): scale parameter (:math:`s`) where scale >= 0
        loc(float): loc parameter (:math:`m`)
        randvar(float): random variable where x > loc

    Reference:
        .. [#] Wikipedia contributors. (2020, December 7). Fréchet distribution. https://en.wikipedia.org/w/index.php?title=Fr%C3%A9chet_distribution&oldid=992938143
    """

    def __init__(self,  shape: float, scale: float, loc: float):
        if shape < 0 or scale < 0:
            raise ValueError(
                'the value of scale and shape are expected to be greater than 0.')

        self.shape = shape
        self.scale = scale
        self.loc = loc

    def pdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]:
        """
        Args:
            x (Union[List[float], numpy.ndarray, float]): random variable(s)

        Raises:
            ValueError: when there exist a random variate less than or equal to loc parameter

        Returns:
            Union[float, numpy.ndarray]: evaluation of pdf at x
        """
        a = self.shape
        s = self.scale
        m = self.loc

        if isinstance(x, (np.ndarray, List)):
            if not type(x) is np.ndarray:
                x = np.array(x)
            if np.any(x <= m):
                raise ValueError(
                    f'random variables are expected to be greater than {m} -- the loc parameter')
            return (a/s) * np.power((x-m)/s, -1-a)*np.exp(-np.power((x-m)/s, -a))

        if x < m:
            raise ValueError(
                f'random variables are expected to be greater than {m} -- the loc parameter')
        return (a/s) * pow((x-m)/s, -1-a)*m.exp(-pow((x-m)/s, -a))

    def cdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]:
        """
        Args:
            x (Union[List[float], numpy.ndarray, float]): data point(s) of interest

        Returns:
            Union[float, numpy.ndarray]: evaluation of pdf at x
        """
        a = self.shape
        s = self.scale
        m = self.loc

        if isinstance(x, (np.ndarray, List)):
            if not type(x) is np.ndarray:
                x = np.array(x)
            return np.exp(-np.power((x-m)/s, -a))

        return m.exp(-pow((x-m)/s, -a))

    def mean(self) -> float:
        """
        Returns: Mean of the Fréchet distribution.
        """
        if self.shape > 1:
            return self.loc + (self.scale*ss.gamma(1 - 1/self.shape))
        return np.inf

    def median(self) -> float:
        """
        Returns: Median of the Fréchet distribution.
        """
        return self.loc + (self.scale/pow(m.log(2), 1/self.shape))

    def mode(self) -> float:
        """
        Returns: Mode of the Fréchet distribution.
        """
        return self.loc + self.scale*(self.shape/pow(1 + self.shape, 1/self.shape))

    def var(self) -> Union[float, str]:
        """
        Returns: Variance of the Fréchet distribution.
        """
        a = self.shape
        s = self.scale
        if a > 2:
            return pow(s, 2)*(ss.gamma(1-2/a)-pow(ss.gamma(1-1/a), 2))
        return "infinity"

    def std(self) -> Union[float, str]:
        """
        Returns: Standard devtiation of the Fréchet distribution.
        """
        var = self.var()
        if type(var) is float:
            return m.sqrt(var)
        return "infinity"

    def skewness(self) -> Union[float, str]:
        """
        Returns: Skewness of the Fréchet distribution.
        """
        a = self.shape
        if a > 3:
            return (ss.gamma(1-3/a)-3*ss.gamma(1-2/a)*ss.gamma(1-1/a)+2*ss.gamma(1-1/a)**3)/pow(ss.gamma(1-2/a)-pow(ss.gamma(1-1/a), 2), 3/2)
        return "infinity"

    def kurtosis(self) -> Union[float, str]:
        """
        Returns: Kurtosis of the Fréchet distribution.
        """
        a = self.shape
        if a > 4:
            return -6+(ss.gamma(1-4/a)-4*ss.gamma(1-3/a)*ss.gamma(1-1/a)+3*pow(ss.gamma(1-2/a), 2))/pow(ss.gamma(1-2/a)-pow(ss.gamma(1-1/a), 2), 2)
        return "infinity"

    def summary(self) -> Dict[str, Union[float, int, str]]:
        """
        Returns:
            Dictionary of Fréchet distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }


class Gamma(SemiInfinite):
    """
    This class contains methods concerning a variant of Gamma distribution [#]_.

    .. math:: 
        \\text{Gamma}(x;a,b) = \\frac{1}{b^a \\Gamma(a)} \\ x^{a-1} e^{\\frac{-x}{b}}

    Args:

        shape(float): shape parameter (:math:`a`) where shape > 0
        scale(float): scale parameter (:math:`b`) where scale > 0
        x(float): random variable where x > 0

    References:
        .. [#] Matlab(2020). Gamma Distribution. https://www.mathworks.com/help/stats/gamma-distribution.html
    """

    def __init__(self, shape: float, b: float, x: float):
        if shape < 0:
            raise ValueError('shape should be greater than 0.')
        if b < 0:
            raise ValueError('scale should be greater than 0.')
        self.shape = shape
        self.scale = b

    def pdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]:
        """
        Args:
            x (Union[List[float], numpy.ndarray, float]): random variable(s)

        Raises:
            ValueError: when there exist a value of x that is less than 0

        Returns:
            Union[float, numpy.ndarray]: evaluation of pdf at x
        """
        shape = self.shape
        scale = self.scale

        if isinstance(x, (np.ndarray, List)):
            if not type(x) is np.ndarray:
                x = np.array(x)
            if np.any(x < 0):
                raise ValueError('random variable should be greater than 0.')
            return (1 / (pow(scale, shape) * ss.gamma(shape))) * np.log(x, shape - 1) * np.exp(-x / scale)

        if x < 0:
            raise ValueError('random variable should be greater than 0.')
        return (1 / (pow(scale, shape) * ss.gamma(shape))) * m.log(x, shape - 1) * m.exp(-x / scale)

    def cdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]:
        """
        Args:
            x (Union[List[float], numpy.ndarray, float]): data point(s) of interest

        Returns:
            Union[float, numpy.ndarray]: evaluation of cdf at x
        """
        shape = self.shape
        scale = self.scale

        # there is no apparent explanation for reversing gammainc's parameter, but it works quite perfectly in my prototype
        def __generator(shape: float, b: float, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
            return 1 - ss.gammainc(shape, x / b)

        if isinstance(x, (np.ndarray, List)):
            if not type(x) is np.ndarray:
                x = np.array(x)
            return __generator(shape, scale, x)
        return __generator(shape, scale, x)

    def mean(self) -> Union[float, int]:
        """
        Returns: Mean of the Gamma distribution
        """
        return self.shape * self.scale

    def median(self) -> str:
        """
        Returns: Median of the Gamma distribution.
        """
        return "No simple closed form."

    def mode(self) -> Union[float, int]:
        """
        Returns: Mode of the Gamma distribution
        """
        return (self.shape - 1) * self.scale

    def var(self) -> Union[float, int]:
        """
        Returns: Variance of the Gamma distribution
        """
        return self.shape * pow(self.scale, 2)

    def std(self) -> float:
        """
        Returns: Standard deviation of the Gamma distribution
        """
        return m.sqrt(self.var())

    def skewness(self) -> float:
        """
        Returns: Skewness of the Gamma distribution
        """
        return 2 / m.sqrt(self.shape)

    def kurtosis(self) -> float:
        """
        Returns: Kurtosis of the Gamma distribution
        """
        return 6 / self.shape

    def entropy(self) -> float:
        """
        Returns: differential entropy of the Gamma distribution

        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier.
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        k = self.shape
        theta = self.scale
        return k + m.log(theta)+m.log(ss.gamma(k))-(1-k)*ss.digamma(k)

    def summary(self) -> Dict[str, Union[float, int, str]]:
        """
        Returns:
            Dictionary of Gamma distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }


class F(SemiInfinite):
    """
    This class contains methods concerning the F-distribution [#]_ [#]_ [#]_.

    .. math::
        \\text{F}(x;d_1, d_2) = \\frac{1}{\\text{B}(d_1/2,d_2/2)} \\Big( \\frac{d_1}{d_2} \\Big)^{d_1/2} x^{d_1/2 - 1} \\Big(1 + \\frac{d_1}{d_2}x\\Big) ^{-(d_1+d_2)/2}

    Args:

        df1(int): first degrees of freedom where df1 > 0
        df2(int): second degrees of freedom where df2 > 0 
        x(float): random variable where x > 0

    References:
        .. [#] Mood, Alexander; Franklin A. Graybill; Duane C. Boes (1974). Introduction to the Theory of Statistics (Third ed.). McGraw-Hill. pp. 246–249. ISBN 0-07-042864-6.
        .. [#] Weisstein, Eric W. "F-Distribution." From MathWorld--A Wolfram Web Resource. https://mathworld.wolfram.com/F-Distribution.html
        .. [#] NIST SemaTech (n.d.). F-Distribution. Retrived from https://www.itl.nist.gov/div898/handbook/eda/section3/eda3665.htm
    """

    def __init__(self, df1: int, df2: int):
        if (type(df1) is not int) or (df1 <= 0):
            raise TypeError(
                f'degrees of freedom(df) should be a whole number.')
        if (type(df1) is not int) or (df2 <= 0):
            raise TypeError(
                f'degrees of freedom(df) should be a whole number.')

        self.df1 = df1
        self.df2 = df2

    def pdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]:
        """
        Args:
            x (Union[List[float], numpy.ndarray, float]): random variable(s)

        Raises:
            ValueError: when there exist a value such that x < 0

        Returns:
            Union[float, numpy.ndarray]: evaluation of pdf at x
        """

        def __generator(x: Union[float, np.ndarray], df1: int, df2: int) -> Union[float, np.ndarray]:
            x0 = (1 / ss.beta(df1 / 2, df2 / 2)) * pow(df1 / df2, df1 / 2)
            if type(x) is np.ndarray:
                return x0 * \
                    np.power(x, df1 / 2 - 1) * np.power(1 +
                                                          (df1 / df2) * x, -((df1 + df2) / 2))
            return x0 * \
                pow(x, df1 / 2 - 1) * \
                pow(1 + (df1 / df2) * x, -((df1 + df2) / 2))

        if isinstance(x, (np.ndarray, List)):
            if not type(x) is np.ndarray:
                x = np.array(x)
            if np.any(x < 0):
                raise ValueError(
                    'random variables are expected to be greater than 0.')
            return __generator(x, self.df1, self.df2)

        if x < 0:
            raise ValueError(
                'random variable is expected to be greater than 0.')
        return __generator(x, self.df1, self.df2)

    def cdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]:
        """
        Args:
            x (Union[List[float], numpy.ndarray, float]): data point(s)

        Returns:
            Union[float, numpy.ndarray]: evaluates cdf at x
        """

        df1, df2 = self.df1, self.df2

        if isinstance(x, (np.ndarray, List)):
            if not type(x) is np.ndarray:
                x = np.array(x)
            return ss.betainc(df1/2, df2/2, df1*x/(df1*x + df2))

        k = df1*x/(self.df2 + self.df1*x)
        return ss.betainc(df1/2, df2/2, k)

    def mean(self) -> Union[float, str]:
        """
        Returns: Mean of the F-distribution.
        """
        if self.df2 > 2:
            return self.df2 / (self.df2 - 2)
        return "undefined"

    def mode(self) -> Union[float, str]:
        """
        Returns: Mode of the F-distribution.
        """
        df1 = self.df1
        df2 = self.df2
        if df1 > 2:
            return (df2 * (df1 - 2)) / (df1 * (df2 + 2))
        return "undefined"

    def var(self) -> Union[float, str]:
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
            return m.sqrt((2 * pow(df2, 2) * (df1 + df2 - 2))/(df1 * (pow(df2 - 2, 2) * (df2 - 4))))
        return 'undefined'

    def skewness(self) -> Union[float, str]:
        """
        Returns: Skewness of the F-distribution.
        """
        df1 = self.df1
        df2 = self.df2
        if df2 > 6:
            return ((2 * df1 + df2 - 2) * m.sqrt(8 * (df2 - 4))) / ((df2 - 6) * m.sqrt(df1 * (df1 + df2 - 2)))
        return "undefined"

    def entropy(self) -> Union[float, int]:
        """
        Returns: differential entropy of F-distribution.

        Reference: Lazo, A.V.; Rathie, P. (1978). "On the entropy of continuous probability distributions". IEEE Transactions on Information Theory
        """
        df1 = self.df1
        df2 = self.df2
        return m.log(ss.gamma(df1/2)) + m.log(ss.gamma(df2/2)) -\
            m.log(ss.gamma((df1+df2)/2)) + (1-df1/2)*ss.digamma(1+df1/2) -\
            (1-df2/2) * ss.digamma(1+df2/2) + (df1+df2) /\
            2*ss.digamma((df1+df2)/2) + m.log(df1/df2)

    def summary(self) -> Dict[str, Union[float, str]]:
        """
        Returns:
            Dictionary of F distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }


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

    def __init__(self, df: int):
        if type(df) is not int:
            raise TypeError('degrees of freedom(df) should be a whole number.')

        if df <= 0:
            raise ValueError('df parameter must be a positive integer.')

        self.df = df

    def pdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]:
        """
        Args:
            x (Union[List[float], numpy.ndarray, float]): random variable(s)

        Returns:
            Union[float, numpy.ndarray]: evaluation of pdf at x
        """
        df = self.df

        if isinstance(x, (np.ndarray, List)):
            if not type(x) is np.ndarray:
                x = np.array(x)
            return (1 / (pow(2, (df / 2) - 1) * ss.gamma(df / 2))) * np.power(x, df - 1) * np.exp(np.power(-x, 2) / 2)

        return (1 / (pow(2, (df / 2) - 1) * ss.gamma(df / 2))) * pow(x, df - 1) * np.exp(pow(-x, 2) / 2)

    def cdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]:
        """
        Args:
            x (Union[List[float], numpy.ndarray]): data point(s) of interest

        Returns:
            Union[float, numpy.ndarray]: evaluation of cdf at x
        """
        df = self.df

        if isinstance(x, (np.ndarray, List)):
            if not type(x) is np.ndarray:
                x = np.array(x)
            return ss.gammainc(df/2, np.power(x, 2)/2)

        return ss.gammainc(df/2, pow(x, 2)/2)

    def mean(self) -> float:
        """
        Returns: Mean of the Chi distribution.
        """
        return m.sqrt(2)*ss.gamma((self.df+1)/2)/ss.gamma(self.df/2)

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
            return m.sqrt(self.df-1)
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

        return 2*(1-mean*m.sqrt(var)*sk-var)/var

    def entropy(self) -> float:
        """
        Returns: differential entropy of Chi distribution.

        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier.
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        df = self.df
        return m.log(ss.gamma(df/2)/m.sqrt(2)) - (df-1)/2*ss.digamma(df/2) + df/2

    def summary(self) -> Dict[str, Union[float, int, str]]:
        """
        Returns:
            Dictionary of Chi distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }


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

    def pdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]:
        """
        Args:
            x (Union[List[float], numpy.ndarray, float]): random variable(s)

        Raises:
            ValueError: when there exist a vaue less than 0

        Returns:
            Union[float, numpy.ndarray]: evaluation of pdf at x
        """
        df = self.df

        if isinstance(x, (np.ndarray, List)):
            if not type(x) is np.ndarray:
                x = np.array(x)
            if np.any(x < 0):
                raise ValueError(
                    'random variables are only valid for positive real numbers')
            return (1 / (np.power(2, (df / 2) - 1) * ss.gamma(df / 2))) * np.power(x, df - 1) * np.exp(-np.power(x, 2) / 2)

        if x < 0:
            raise ValueError(
                'random variable are only valid for positive real numbers')
        return (1 / (pow(2, (df / 2) - 1) * ss.gamma(df / 2))) * pow(x, df - 1) * np.exp(-pow(x, 2) / 2)

    def cdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]:
        """
        Args:
            x (Union[List[float], numpy.ndarray, float]): data point(s) of interest

        Raises:
            ValueError: when there exist a value of x less than 0

        Returns:
            Union[float, numpy.ndarray]: evaluation of cdf at x
        """
        df = self.df

        if isinstance(x, (np.ndarray, List)):
            if not type(x) is np.ndarray:
                x = np.array(x)
            if np.any(x < 0):
                raise ValueError(
                    'data point(s) are only valid for positive real numbers')
            return ss.gammainc(df/2, x/2)

        if x < 0:
            raise ValueError(
                'data point(s) are only valid for positive real numbers')
        return ss.gammainc(df/2, x/2)

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
        return m.sqrt(2 * self.df)

    def skewness(self) -> float:
        """
        Returns: Skewness of the Chi-square distribution.
        """
        return m.sqrt(8 / self.df)

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
        return df/2 + m.log(2*ss.gamma(df/2)) + (1-df/2)*ss.digamma(df/2)

    def summary(self) -> Dict[str, Union[float, str]]:
        """
        Returns:
            Dictionary of Chi-Square distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }


class Erlang(SemiInfinite):
    """
    This class contains methods concerning Erlang Distirbution [#]_ [#]_.

    .. math:: 
        \\text{Erlang}(x; k, \\lambda) = \\frac{\\lambda^{k} x^{k-1} e^{\\lambda x}}{(k-1)!}

    Args:

        shape(int): shape parameter (:math:`k`) where shape > 0
        rate(float): rate parameter (:math:`\\lambda`) where rate >= 0
        x(float): random variable where x >= 0

    Reference:
        .. [#] Wikipedia contributors. (2021, January 6). Erlang distribution. https://en.wikipedia.org/w/index.php?title=Erlang_distribution&oldid=998655107
        .. [#] Weisstein, Eric W. "Erlang Distribution." From MathWorld--A Wolfram Web Resource. https://mathworld.wolfram.com/ErlangDistribution.html
    """

    def __init__(self, shape: int, rate: float):
        if type(shape) is not int and shape > 0:
            raise TypeError(
                'shape parameter should be an integer greater than 0.')
        if rate < 0:
            raise ValueError(
                f'beta parameter(rate) should be a positive number.')

        self.shape = shape
        self.rate = rate

    def pdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]:
        """
        Args:
            x (Union[List[float], numpy.ndarray, float]): random variable(s)

        Raises:
            ValueError: when there exist a value of x that is less than 0 or greater than 1

        Returns:
            Union[float, numpy.ndarray]: evaluation of pdf at x
        """
        shape = self.shape
        rate = self.rate

        if isinstance(x, (np.ndarray, List)):
            if not type(x) is np.ndarray:
                x = np.array(x)
            if np.any((x < 0)|(x > 1)):
                raise ValueError(
                    'random variable should only be in between 0 and 1')
            return pow(rate, shape) * np.power(x, shape-1)*np.exp(-rate*x) / m.factorial(shape-1)

        if x < 0 or x > 1:
            raise ValueError(
                'random variable should only be in between 0 and 1')
        return pow(rate, shape)*pow(x, shape-1)*m.exp(-rate*x)/m.factorial(shape-1)

    def cdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]:
        """
        Args:
            x (Union[List[float], numpy.ndarray, float]): data point(s) of interest

        Raises:
            ValueError: when there exist a data value of x that is less than 0 or greater than 1

        Returns:
            Union[float, numpy.ndarray]: evaluation of cdf at x
        """
        shape = self.shape
        rate = self.rate

        if isinstance(x, (np.ndarray, List)):
            if not type(x) is np.ndarray:
                x = np.array(x)
            if np.any((x < 0)|(x > 1)):
                raise ValueError(
                    'random variable should only be in between 0 and 1')
            return ss.gammainc(shape, rate*x)/m.factorial(shape-1)

        if x < 0 or x > 1:
            raise ValueError(
                'random variable should only be in between 0 and 1')
        return ss.gammainc(shape, rate*x)/m.factorial(shape-1)

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
        return m.sqrt(self.var())

    def skewness(self) -> float:
        """
        Returns: Skewness of the Erlang distribution.
        """
        return 2/m.sqrt(self.shape)

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
        lmbda = self.rate
        return (1-k)*ss.digamma(k)+m.log(ss.gamma(k)/lmbda)+k

    def summary(self) -> Dict[str, Union[float, int, str]]:
        """
        Returns:
            Dictionary of Erlang distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }


class Rayleigh(SemiInfinite):
    """
    This class contains methods concerning Rayleigh Distirbution [#]_ [#]_.

    .. math:: \\text{Rayleigh}(x;\\sigma) = \\frac{x}{\\sigma^2} \\exp{-(x^2/(2\\sigma^2))}

    Args:

        scale(float): scale parameter (:math:`\\sigma`) where scale > 0
        x(float): random variable where x >= 0

    Reference:
        .. [#] Wikipedia contributors. (2020, December 30). Rayleigh distribution. https://en.wikipedia.org/w/index.php?title=Rayleigh_distribution&oldid=997166230
        .. [#] Weisstein, Eric W. "Rayleigh Distribution." From MathWorld--A Wolfram Web Resource. https://mathworld.wolfram.com/RayleighDistribution.html
    """

    def __init__(self, scale: float):
        if scale < 0:
            raise ValueError('scale parameter should be a positive number.')

        self.scale = scale

    def pdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]:
        """
        Args:
            x (Union[List[float], numpy.ndarray, float]): random variable(s)

        Raises:
            ValueError: when there exist a value of x that is less than 0

        Returns:
            Union[float, numpy.ndarray]: evaluation of pdf at x
        """
        sig = self.scale  # scale to sig

        if isinstance(x, (np.ndarray, List)):
            if not type(x) is np.ndarray:
                x = np.array(x)
            if np.any(x < 0):
                raise ValueError('random variable must be a positive number')
            return x/pow(sig, 2) * np.exp(np.power(-x, 2)/(2*pow(sig, 2)))

        if x < 0:
            raise ValueError('random variable must be a positive number')
        return x/pow(sig, 2) * m.exp(pow(-x, 2)/(2*pow(sig, 2)))

    def cdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]:
        """
        Args:
            x (Union[List[float], numpy.ndarray, float]): data point(s) of interest

        Returns:
            Union[float, numpy.ndarray]: evaluation of cdf at x
        """
        sig = self.scale

        if isinstance(x, (np.ndarray, List)):
            if not type(x) is np.ndarray:
                x = np.array(x)
            return 1-np.exp(-np.power(x, 2)/(2*sig**2))

        return 1-m.exp(-x**2/(2*sig**2))

    def mean(self) -> float:
        """
        Returns: Mean of the Rayleigh distribution.
        """
        return self.scale*m.sqrt(m.pi/2)

    def median(self) -> float:
        """
        Returns: Median of the Rayleigh distribution.
        """
        return self.scale*m.sqrt(2*m.log(2))

    def mode(self) -> float:
        """
        Returns: Mode of the Rayleigh distribution.
        """
        return self.scale

    def var(self) -> float:
        """
        Returns: Variance of the Rayleigh distribution.
        """
        return (4-m.pi)/2*pow(self.scale, 2)

    def std(self) -> float:
        """
        Returns: Standard deviation of the Rayleigh distribution
        """
        return m.sqrt((4-m.pi)/2*pow(self.scale, 2))

    def skewness(self) -> float:
        """
        Returns: Skewness of the Rayleigh distribution.
        """
        return (2*m.sqrt(m.pi)*(m.pi-3))/pow((4-m.pi), 3/2)

    def kurtosis(self) -> float:
        """
        Returns: Kurtosis of the Rayleigh distribution.
        """
        return -(6*pow(m.pi, 2)-24*m.pi+16)/pow(4-m.pi, *2)

    def entropy(self) -> float:
        """
        Returns: differential entropy of the Rayleigh distribution.

        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier.
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        return 1+m.log(self.scale/m.sqrt(2))+(euler/2)

    def summary(self) -> Dict[str, Union[float, Tuple[float]]]:
        """
        Returns:
            Dictionary of Rayleigh distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }


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
            raise ValueError('scale should be greater than 0.')
        if shape < 0:
            raise ValueError('shape should be greater than 0.')
        if x > shape:
            raise ValueError(
                'random variable x should be greater than or equal to shape.')

        self.shape = shape
        self.scale = scale

    def pdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]:
        """
        Args:
            x (Union[List[float], numpy.ndarray, float]): random variable(s)

        Raises:
            ValueError: when there is a case that a random variable is greater than the value of shape parameter

        Returns:
            Union[float, numpy.ndarray]: evaluation of pdf at x
        """
        x_m = self.scale
        alpha = self.shape

        if isinstance(x, (np.ndarray, List)):
            if not type(x) is np.ndarray:
                x = np.array(x)
            if np.any(x > alpha):
                raise ValueError(
                    'random variable should be greater thaan or equal to the value of shape')
            return np.piecewise(x, [x >= x_m, x < x_m], [lambda x: alpha*np.power(x_m, alpha)/np.power(x, alpha + 1), lambda x: 0.0])

        if x > alpha:
            raise ValueError(
                'random variable should be greater thaan or equal to the value of shape')
        return alpha*np.power(x_m, alpha)/np.power(x, alpha + 1) if x >= x_m else 0.0

    def cdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]:
        """

        Args:
            x (Union[List[float], numpy.ndarray, float]): data point(s) of interest

        Returns:
            Union[float, numpy.ndarray]: evaluation of cdf at x
        """
        x_m = self.scale
        alpha = self.shape

        if isinstance(x, (np.ndarray, List)):
            if not type(x) is np.ndarray:
                x = np.array(x)
            return np.piecewise(x, [x >= x_m, x < x_m], [lambda x: 1 - np.power(x_m/x, alpha), lambda x: 0.0])

        return 1 - pow(x_m/x, alpha) if x >= x_m else 0.0

    def mean(self) -> float:
        """
        Returns: Mean of the Pareto distribution.
        """
        a = self.shape
        x_m = self.scale

        if a <= 1:
            return np.inf
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
            return np.inf
        return (pow(x_m, 2) * a) / (pow(a - 1, 2) * (a - 2))

    def std(self) -> float:
        """
        Returns: Variance of the Pareto distribution
        """
        return m.sqrt(self.var())

    def skewness(self) -> Union[float, str]:
        """
        Returns: Skewness of the Pareto distribution.
        """
        a = self.shape
        x_m = self.scale
        if a > 3:
            scale = (2 * (1 + a)) / (a - 3)
            return scale * m.sqrt((a - 2) / a)
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
        return m.log(x_m/a)+1+(1/a)

    def summary(self) -> Dict[str, Union[float, str]]:
        """
        Returns:
            Dictionary of Pareto distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }


class MaxwellBoltzmann(SemiInfinite):
    """
    This class contains methods concerning Maxwell-Boltzmann Distirbution [#]_.

    .. math::
        \\text{MaxwellBoltzmann}(x;a) = \\sqrt{\\frac{2}{\\pi}} \\frac{x^2 \\exp{-x^2/(2a^2)}}{a^3}

    Args:

        a(int): parameter where a > 0
        x(float): random variable where x >= 0

    Reference:
        .. [#] Wikipedia contributors. (2021, January 12). Maxwell–Boltzmann distribution. https://en.wikipedia.org/w/index.php?title=Maxwell%E2%80%93Boltzmann_distribution&oldid=999883013
    """

    def __init__(self, a: int):
        if a < 0:
            raise ValueError(
                'parameter a should be a positive number. Entered value:{}'.format(a))
        if type(a) is not int:
            raise TypeError('parameter should be in type int')

        self.a = a

    def pdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]:
        """
        Args:
            x (Union[List[float], numpy.ndarray, float]): random variable(s)

        Raises:
            ValueError: when there exist a value of x less than 0

        Returns:
            Union[float, numpy.ndarray]: evaluation of pdf at x
        """
        a = self.a

        if isinstance(x, (np.ndarray, List)):
            if not type(x) is np.ndarray:
                x = np.array(x)
            if np.any(x < 0):
                raise ValueError('random values must not be lesser than 0')
            return m.sqrt(2/m.pi)*(x**2*np.exp(-x**2/(2*a**2)))/a**3

        if x < 0:
            raise ValueError('random values must not be lesser than 0')
        return m.sqrt(2/m.pi)*(x**2*m.exp(-x**2/(2*a**2)))/a**3

    def cdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]:
        """
        Args:
            x (Union[List[float], numpy.ndarray, float]): data point(s) or interest

        Returns:
            Union[float, numpy.ndarray]: evaluation of cdf at x
        """
        a = self.a

        if isinstance(x, (np.ndarray, List)):
            if not type(x) is np.ndarray:
                x = np.array(x)
            x0 = np.power(x, 2)
            return ss.erf(x/(m.sqrt(2)*a))-m.sqrt(2/m.pi)*(x0*np.exp(-x0/(2*a**2)))/(a)

        return ss.erf(x/(m.sqrt(2)*a)) - m.sqrt(2/m.pi)*(x**2*m.exp(-x**2/(2*a**2)))/(a)

    def mean(self) -> float:
        """
        Returns: Mean of the Maxwell-Boltzmann distribution.
        """
        return 2*self.a*m.sqrt(2/m.pi)

    def median(self) -> Union[float, str]:
        """
        Returns: Median of the Maxwell-Boltzmann distribution.
        """
        return "currently unsupported"

    def mode(self) -> float:
        """
        Returns: Mode of the Maxwell-Boltzmann distribution.
        """
        return m.sqrt(2)*self.a

    def var(self) -> float:
        """
        Returns: Variance of the Maxwell-Boltzmann distribution.
        """
        return (self.a**2*(3*m.pi-8))/m.pi

    def std(self) -> float:
        """
        Returns: Standard deviation of the Maxwell-Boltzmann distribution
        """
        return m.sqrt((self.a**2*(3*m.pi-8))/m.pi)

    def skewness(self) -> float:
        """
        Returns: Skewness of the Maxwell-Boltzmann distribution.
        """
        return (2*m.sqrt(2)*(16-5*m.pi))/np.power((3*m.pi-8), 3/2)

    def kurtosis(self) -> float:
        """
        Returns: Kurtosis of the Maxwell-Boltzmann distribution.
        """
        return 4*((-96+40*m.pi-3*m.pi**2)/(3*m.pi-8)**2)

    def entropy(self) -> float:
        """
        Returns: differential entropy of the Maxwell-Boltzmann distribution.

        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier.
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        a = self.a
        return m.log(a*m.sqrt(2*m.pi)+euler-0.5)

    def summary(self) -> Dict[str, Union[float, str]]:
        """
        Returns:
            Dictionary of Maxwell-Boltzmann distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }


class LogNormal(SemiInfinite):
    """
    This class contains methods concerning the Log Normal Distribution [#]_ [#]_.

    .. math::
        \\text{LogNormal}(x;\\mu,\\sigma) = \\frac{1}{x\\sigma\\sqrt{2\\pi}} \\exp{\\Big( - \\frac{(\\ln x - \\mu)^2}{2\\sigma^2} \\Big)}

    Args:

        mean (float): mean parameter (:math:`\\mu`)
        std (float): standard deviation (:math:`\\sigma`) where std > 0
        x (float): random variable where x >= 0

    References:
        .. [#] Weisstein, Eric W. "Log Normal Distribution." From MathWorld--A Wolfram Web Resource.https://mathworld.wolfram.com/LogNormalDistribution.html
        .. [#] Wikipedia contributors. (2020, December 18). Log-normal distribution. https://en.wikipedia.org/w/index.php?title=Log-normal_distribution&oldid=994919804
    """

    def __init__(self, mean: float, std: float, randvar: float):
        if randvar < 0:
            raise ValueError('random variable should be greater than 0.')
        if std < 0:
            raise ValueError('random variable should be greater than 0.')

        self.randvar = randvar
        self.mean_val = mean
        self.stdev = std

    def pdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]:
        """
        Args:
            x (Union[List[float], numpy.ndarray, float]): random variable(s)

        Raises:
            ValueError: when there exist a value of x < 0

        Returns:
            Union[float, numpy.ndarray]: evaluation of pdf at x
        """
        mean = self.mean_val
        stdev = self.stdev

        if isinstance(x, (np.ndarray, List)):
            if not type(x) is np.ndarray:
                x = np.array(x)
            if np.any(x < 0):
                raise ValueError('random variable should be greater than 0.')
            return 1 / (x * stdev * m.sqrt(2 * m.pi)) * np.exp(-(np.log(x - mean)**2) / (2 * stdev**2))

        if x < 0:
            raise ValueError('random variable should be greater than 0.')
        return 1 / (x * stdev * m.sqrt(2 * m.pi)) * m.exp(-(m.log(x - mean)**2) / (2 * stdev**2))

    def cdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]:
        """
        Args:
            x (Union[List[float], numpy.ndarray, float]): data point(s) of interest

        Returns:
            Union[float, numpy.ndarray]: evaluation of cdf at x
        """
        mean = self.mean_val
        std = self.stdev

        if isinstance(x, (np.ndarray, List)):
            if not type(x) is np.ndarray:
                x = np.array(x)
            return 0.5 + 0.5*ss.erfc(-np.log(x - mean)/(std * m.sqrt(2)))

        return 0.5 + 0.5*ss.erfc(-np.log(x - mean)/(std * m.sqrt(2)))

    def mean(self) -> float:
        """
        Returns: Mean of the log normal distribution.
        """
        return m.exp(self.mean_val + pow(self.stdev, 2) / 2)

    def median(self) -> float:
        """
        Returns: Median of the log normal distribution.
        """
        return m.exp(self.mean_val)

    def mode(self) -> float:
        """
        Returns: Mode of the log normal distribution.
        """
        return m.exp(self.mean_val - pow(self.stdev, 2))

    def var(self) -> float:
        """
        Returns: Variance of the log normal distribution.
        """
        std = self.stdev
        mean = self.mean_val
        return (m.exp(pow(std, 2)) - 1) * m.exp(2 * mean + pow(std, 2))

    def std(self) -> float:
        """
        Returns: Standard deviation of the log normal distribution
        """
        return self.stdev

    def skewness(self) -> float:
        """
        Returns: Skewness of the log normal distribution.
        """
        std = self.stdev
        return (m.exp(pow(std, 2)) + 2) * m.sqrt(m.exp(pow(std, 2)) - 1)

    def kurtosis(self) -> float:
        """
        Returns: Kurtosis of the log normal distribution.
        """
        std = self.stdev
        return m.exp(
            4 * pow(std, 2)) + 2 * m.exp(3 * pow(std, 2)) + 3 * m.exp(2 * pow(std, 2)) - 6

    def entropy(self) -> float:
        """
        Returns: differential entropy of the log normal distribution.

        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier.
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        return self.mean_val + 0.5 * m.log(2*m.pi*m.e*self.stdev**2)

    def summary(self) -> Dict[str, Union[float, int, str]]:
        """
        Returns:
            Dictionary of Log Normal distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }


class BetaPrime(SemiInfinite):
    """
    This class contains methods concerning Beta prime Distirbution [#]_ .

    .. math:: 
        \\text{BetaPrime}(x;\\alpha,\\beta) = \\frac{x^{\\alpha -1}(1+x)^{-\\alpha -\\beta}}{\\text{B}(\\alpha ,\\beta )}

    Args:

        alpha(float): shape parameter where alpha > 0
        beta(float): shape parameter where beta > 0
        x(float): random variable where x >= 0

    Reference:
        .. [#] Wikipedia contributors. (2020, October 8). Beta prime distribution. https://en.wikipedia.org/w/index.php?title=Beta_prime_distribution&oldid=982458594
    """

    def __init__(self, alpha: float, beta: float):
        if alpha < 0:
            raise ValueError(
                'alpha parameter(shape) should be a positive number.')
        if beta < 0:
            raise ValueError(
                'beta parameter(shape) should be a positive number.')

        self.alpha = alpha
        self.beta = beta

    def pdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]:
        """
        Args:
            x (Union[List[float], numpy.ndarray, float]): random variable(s)

        Raises:
            ValueError: when there exist a value of x less than 0

        Returns:
            Union[float, numpy.ndarray]: evaluation of pdf at x
        """
        a = self.alpha
        b = self.beta

        if isinstance(x, (np.ndarray, List)):
            if not type(x) is np.ndarray:
                x = np.array(x)
            if np.any(x < 0):
                raise ValueError('random variable should not be less then 0.')
            return np.power(x, a-1)*np.power(1+x, -a-b)/ss.beta(a, b)

        if x < 0:
            raise ValueError('random variable should not be less then 0.')
        return pow(x, a-1)*pow(1+x, -a-b)/ss.beta(a, b)

    def cdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]:
        """
        Args:
            x (Union[List[float], numpy.ndarray, float]): data point(s) of interest

        Raises:
            ValueError: when there exist a value of x less than 0

        Returns:
            Union[float, numpy.ndarray]: evaluation of cdf at x
        """
        a = self.alpha
        b = self.beta

        if isinstance(x, (np.ndarray, List)):
            if not type(x) is np.ndarray:
                x = np.array(x)
            if np.any(x < 0):
                raise ValueError(
                    'evaluation of cdf is not supported for values less than 0')
            return ss.betainc(a, b, x/(1+x))

        return ss.betainc(a, b, x/(1+x))

    def mean(self) -> Union[float, str]:
        """
        Returns: Mean of the Beta prime distribution.
        """
        if self.beta > 1:
            return self.alpha/(self.beta-1)
        return "Undefined."

    def median(self) -> str:
        """
        Returns: Median of the Beta prime distribution.
        """
        # warning: not yet validated.
        return "Undefined."

    def mode(self) -> float:
        """
        Returns: Mode of the Beta prime distribution.
        """
        if self.alpha >= 1:
            return (self.alpha+1)/(self.beta+1)
        return 0.0

    def var(self) -> Union[float, str]:
        """
        Returns: Variance of the Beta prime distribution.
        """
        alpha = self.alpha
        beta = self.beta
        if beta > 2:
            return (alpha*(alpha+beta-1))/((beta-2)*(beta-1)**2)
        return "Undefined."

    def std(self) -> Union[float, str]:
        """
        Returns: Standard deviation of the Log logistic distribution
        """
        var = self.var()
        if type(var) is str:
            return "Undefined."
        return m.sqrt(var)

    def skewness(self) -> Union[float, str]:
        """
        Returns: Skewness of the Beta prime distribution.
        """
        alpha = self.alpha
        beta = self.beta
        if beta > 3:
            scale = (2*(2*alpha+beta-1))/(beta-3)
            return scale*m.sqrt((beta-2)/(alpha*(alpha+beta-1)))
        return "Undefined."

    def kurtosis(self) -> str:
        """
        Returns: Kurtosis of the Beta prime distribution.
        """
        return "Undefined."

    def entropy(self):
        """
        Returns: differential entropy of the Beta prime distribution.

        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier.
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        return NotImplemented

    def summary(self) -> Dict[str, Union[float, str]]:
        """
        Returns:
            Dictionary of BetaPrime distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }


class Gumbell(SemiInfinite):
    """
    This class contains methods concerning Gumbel Distirbution [#]_.

    .. math::
        \\text{Gumbel}(x;\\mu,\\beta) = \\frac{1}{\\beta} \\exp{-\\Big( \\frac{x-\\mu}{\\beta} + \\exp{ \\frac{x-\\mu}{\\beta}} \\Big)}

    Args:

        location(float): location parameter (:math:`\\mu`)
        scale(float): scale parameter (:math:`\\beta`) where scale > 0
        x(float): random variable

    Reference:
        .. [#] Wikipedia contributors. (2020, November 26). Gumbel distribution. https://en.wikipedia.org/w/index.php?title=Gumbel_distribution&oldid=990718796
    """

    def __init__(self, location: float, scale: float):
        if scale < 0:
            raise ValueError(
                f'scale parameter should be greater than 0. The value of the scale parameter is: {scale}')

        self.location = location
        self.scale = scale

    def pdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]:
        """

        Args:
            x (Union[List[float], numpy.ndarray, float]): random variable(s)

        Returns:
            Union[float, numpy.ndarray]: evaluation of pdf at x
        """
        mu = self.location
        beta = self.scale

        if isinstance(x, (np.ndarray, List)):
            if not type(x) is np.ndarray:
                x = np.array(x)
            z = (x-mu)/beta
            return (1/beta)*np.exp(-(z+np.exp(-z)))

        z = (x-mu)/beta
        return (1/beta)*m.exp(-(z+m.exp(-z)))

    def cdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]:
        """
        Args:
            x (Union[List[float], numpy.ndarray, float]): data point(s) of interest

        Returns:
            Union[float, numpy.ndarray]: evaluation of cdf at x
        """
        mu = self.location
        beta = self.scale

        if isinstance(x, (np.ndarray, List)):
            if not type(x) is np.ndarray:
                x = np.array(x)
            return np.exp(-np.exp(-(x-mu)/beta))
        return m.exp(-m.exp(-(x - mu)/beta))

    def mean(self) -> float:
        """
        Returns: Mean of the Gumbel distribution.
        """
        return self.location+(self.scale*euler)

    def median(self) -> float:
        """
        Returns: Median of the Gumbel distribution.
        """
        return self.location - (self.scale*m.log(m.log(2)))

    def mode(self) -> float:
        """
        Returns: Mode of the Gumbel distribution.
        """
        return self.location

    def var(self) -> float:
        """
        Returns: Variance of the Gumbel distribution.
        """
        return pow(m.pi, 2/6)*pow(self.scale, 2)

    def std(self) -> float:
        """
        Returns: Standard deviation of the Gumbel distribution.
        """
        return m.sqrt(self.var())

    def skewness(self) -> float:
        """
        Returns: Skewness of the Gumbel distribution.
        """
        return 1.14

    def kurtosis(self) -> float:
        """
        Returns: Kurtosis of the Gumbel distribution.
        """
        return 2.4

    def summary(self) -> Dict[str, float]:
        """
        Returns:
            Dictionary of Gumbel distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }


class Exponential(SemiInfinite):
    """
    This class contans methods for evaluating Exponential Distirbution [#]_ [#]_.

    .. math:: \\text{Exponential}(x;\\lambda) = \\lambda e^{-\\lambda x}

    Args:

        - rate (float): rate parameter (:math:`\\lambda`) where rate > 0
        - x (float): random variable where x > 0

    References:
        .. [#] Weisstein, Eric W. "Exponential Distribution." From MathWorld--A Wolfram Web Resource. https://mathworld.wolfram.com/ExponentialDistribution.html
        .. [#] Wikipedia contributors. (2020, December 17). Exponential distribution. https://en.wikipedia.org/w/index.php?title=Exponential_distribution&oldid=994779060
    """

    def __init__(self, rate: float):
        if rate < 0:
            raise ValueError(f'lambda parameter should be greater than 0.')

        self.rate = rate

    def pdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]:
        """
        Args:
            x (Union[List[float], numpy.ndarray, float]): random variable(s)

        Returns:
            Union[float, numpy.ndarray]: evaluation of pdf at x
        """
        rate = self.rate

        if isinstance(x, (np.ndarray, List)):
            if not type(x) is np.ndarray:
                x = np.array(x)
            return np.piecewise(x, [x >= 0, x < 0], [lambda x: rate*np.exp(-(rate*(x))), lambda x: 0.0])

        return rate*m.exp(-rate*x) if x >= 0 else 0.0

    def cdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]:
        """
        Args:
            x (Union[List[float], numpy.ndarray, float]): data point(s) of interest

        Returns:
            Union[float, numpy.ndarray]: evaluation of cdf at x
        """
        rate = self.rate

        if isinstance(x, (np.ndarray, List)):
            if not type(x) is np.ndarray:
                x = np.array(x)
            return np.piecewise(x, [x > 0, x <= 0], [lambda x: 1 - np.exp(-rate*x), lambda x: 0.0])

        return 1 - m.exp(-rate*x) if x > 0 else 0.0

    def mean(self) -> float:
        """
        Returns: Mean of the Exponential distribution
        """
        return 1 / self.rate

    def median(self) -> float:
        """
        Returns: Median of the Exponential distribution
        """
        return m.log(2) / self.rate

    def mode(self) -> float:
        """
        Returns: Mode of the Exponential distribution
        """
        return 0.0

    def var(self) -> float:
        """
        Returns: Variance of the Exponential distribution
        """
        return 1 / pow(self.rate, 2)

    def std(self) -> float:
        """
        Returns: Standard deviation of the Exponential distribution
        """
        return m.sqrt(self.var())

    def skewness(self) -> float:
        """
        Returns: Skewness of the Exponential distribution
        """
        return 2.0

    def kurtosis(self) -> float:
        """
        Returns: Kurtosis of the Exponential distribution
        """
        return 6.0

    def entorpy(self) -> float:
        """
        Returns: differential entropy of the Exponential distribution

        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier.
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        return 1 - m.log(self.rate)

    def summary(self) -> Dict[str, Union[float, int]]:
        """
        Returns:
            Dictionary of Exponential distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }



# class Benini(SemiInfinite): ...


# class Burr(SemiInfinite): ...


# class Dagum(SemiInfinite): ...


# class Davis(SemiInfinite): ...

