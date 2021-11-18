try:
    import numpy as np
    import scipy.special as ss
    from scipy.integrate import quad
    from typing import Union, Optional, Tuple, Dict, List
    import math as m
    from univariate._base import Base
except Exception as e:
    print(f"some modules are missing {e}")


class Infinite(Base):
    """
    Description:
        Base class for probability tags.
    """

    def __init__(self):
        if type(self) is Infinite:
            raise TypeError('base class cannot be instantiated.')


class Fisher(Infinite):
    """
    This class contains methods concerning Fisher's z-Distribution [#]_.

    .. math:: 
        \\text{Fisher}(x;d_1, d_2) = \\frac{2d_2^{d_1/2} d_2^{d_2/2}}{\\text{B}\\Big(\\frac{d_1}{2}, \\frac{d_2}{2}\\Big)} \\frac{e^{d_1 x}}{(d_1 e^{2x} + d_2)^{(d_1+d_2)/2}}

    Args:

        df1(float): degrees of freedom (:math:`d_1 > 0`).
        df2(float): degrees of freedom (:math:`d_2 > 0`).
        x(float): random variable.

    Note: Fisher's z-distribution is the statistical distribution of half the log of an F-distribution variate:
    z = 1/2*log(F)

    Reference:
        .. [#] Wikipedia contributors. (2020, December 15). Fisher's z-distribution. https://en.wikipedia.org/w/index.php?title=Fisher%27s_z-distribution&oldid=994427156
    """

    def __init__(self, df1: float, df2: float):
        if df1 <= 0 or df2 <= 0:
            raise ValueError('degrees of freedom  are expected to be positive')

        self.df1 = df1
        self.df2 = df2

    def pdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]:

        df1, df2 = self.df1, self.df2

        x0 = (2*pow(df1, df1/2)*pow(df2, df2/2))
        if isinstance(x, (List, np.ndarray)):
            if not type(x) is np.ndarray:
                x = np.array(x)
            x1 = x0*np.exp(df1*x)
            x2 = ss.beta(df1/2, df2/2) * np.power(df1 *
                                                  np.exp(2*x)+df2, (df1+df2)*0.5)
            return x1/x2

        x1 = x0*m.exp(df1*x)
        x2 = ss.beta(df1/2, df2/2) * pow(df1*m.exp(2*x)+df2, (df1+df2)*0.5)
        return x1/x2

    def summary(self) -> Dict[str, float]:
        """
        Returns:
            Dictionary of Fisher z distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }


class AssymetricLaplace(Infinite):
    """
    This class contains methods concerning to Assymetric Laplace Ditribution [#]_.

    .. math:: 
        {\\displaystyle \\text{AssymetricLaplace}(x;m,\\lambda ,\\kappa )={\frac {\\lambda }{\\kappa +1/\\kappa }}{\\begin{cases}\\exp \\left((\\lambda /\\kappa )(x-m)\\right)&{\\text{if }}x<m\\[4pt]\\exp(-\\lambda \\kappa (x-m))&{\\text{if }}x\\geq m\\end{cases}}}

    Args:
        loc (float): location parameter :math:`m`
        sclae (float): scale parameter :math:`\\lambda > 0`
        asym (float): assymetry parameter :math:`\\kappa > 0`
        x (float): random variable

    Reference:
        .. [#] Wikipedia Contributors (2020). Assymetric Laplace Distribution. https://en.wikipedia.org/wiki/Asymmetric_Laplace_distribution
    """

    def __init__(self, loc: float, scale: float, asym: float):
        if asym <= 0 and scale <= 0:
            raise ValueError(
                'Parameter assymetry and scale is expected to be greater than 0.')

        self.loc = loc
        self.scale = scale
        self.asym = asym

    def pdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]:
        """
        Args:
            x (Union[List[float], np.ndarray, float]): random variable(s)

        Returns:
            Union[float, np.ndarray]: evaluation of pdf at x
        """
        l, k, loc = self.scale, self.asym, self.loc

        x0 = l/(k+1/k)
        if isinstance(x, (List, np.ndarray)):
            if not type(x) is np.ndarray:
                x = np.array(x)

            def f01(x): return np.exp(l/k*(x-loc))
            def f02(x): return np.exp(-l*k*(x-loc))
            return x0*np.piecewise(x, [x < loc, x >= loc], [f01, f02])

        def f1(x): return m.exp(l/k*(x-loc))
        def f2(x): return m.exp(-l*k*(x-loc))

        return x0*(f1(x) if x < loc else f2(x))

    def cdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]:
        l, k, loc = self.scale, self.asym, self.loc

        if isinstance(x, (List, np.ndarray)):
            if not type(x) is np.ndarray:
                x = np.array(x)

            def f01(x): return k**2/(1+k**2)*np.exp(l/k*(x-loc))
            def f02(x): return 1 - 1/(1+k**2)*np.exp(-l*k*(x-loc))
            return np.piecewise(x, [x <= loc, x > loc], [f01, f02])

        def f1(x): return k**2/(1+k**2)*m.exp(l/k*(x-loc))
        def f2(x): return 1 - 1/(1+k**2)*np.exp(-l*k*(x-loc))
        return f1(x) if x <= loc else f2(x)

    def mean(self) -> float:

        k = self.asym

        return self.loc + (1+k**2)/(self.scale*k)

    def median(self) -> Optional[float]:

        loc = self.loc
        k = self.asym
        l = self.scale

        if k > 1:
            return loc + k/l*m.log((1+k**2)/(2*k**2))
        if k < 1:
            return loc + 1/(l*k)*m.log((1+k**2)/2)

    def var(self) -> float:
        k = self.asym
        return (1+k**4)/(self.scale**2*k**2)

    def std(self) -> float:
        k = self.asym
        return m.sqrt((1+k**4)/(self.scale**2*k**2))

    def skewness(self) -> float:
        k = self.asym
        return (2*(1-k**6))/pow(k**4+1, 1.5)

    def kurtosis(self) -> float:
        k = self.asym
        return 6*(1+k**8)/pow(1+k**4, 2)

    def entropy(self) -> float:
        k = self.asym
        return m.log(m.e*(1+k**2)/(k*self.scale))

    def summary(self) -> Dict[str, Optional[float]]:
        """
        Returns:
            Dictionary of Assymetric Laplace distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }


class GNV1(Infinite):
    """
    This class contains methods concerning to Generalized Normal Distribution V1 [#]_. 

    Args:
        loc (float): location parameter :math:`\\mean`
        scale (float): scale parameter :math:`\\alpha`
        shape (float): shape parameter :math:`\\beta`
        x (float): random variable
    """

    def __init__(self, loc: float, scale: float, shape: float) -> None:
        self.loc = loc
        self.scale = scale
        self.shape = shape

    def pdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]:
        b, a, mu = self.shape, self.scale, self.loc

        x0 = b/(2*a*ss.gamma(1/b))

        if isinstance(x, (List, np.ndarray)):
            if not type(x) is np.ndarray:
                x = np.array(x)
            return x0*np.exp(np.power(-np.abs(x-mu)/a, 2))
        return x0*m.exp(pow(-abs(x-mu)/a, 2))

    def cdf(self, x: Union[List[float], np.ndarray, float]
            ) -> Union[float, np.ndarray]: ...

    def mean(self) -> float:
        return self.loc

    def median(self) -> float:
        return self.loc

    def mode(self) -> float:
        return self.loc

    def var(self) -> float:
        a, b = self.scale, self.shape
        return a**2*ss.gamma(3/b)/ss.gamma(1/b)

    def std(self) -> float:
        a, b = self.scale, self.shape
        return m.sqrt(a**2*ss.gamma(3/b)/ss.gamma(1/b))

    def skewness(self) -> float:
        return 0.0

    def kurtosis(self) -> float:
        b = self.shape
        return ss.gamma(5/b)*ss.gamma(1/b)/ss.gamma(3/b)**2 - 3

    def entropy(self) -> float:
        b = self.shape
        return 1/b - m.log(b/(2*self.scale*ss.gamma(1/b)))

    def summary(self) -> Dict[str, Optional[float]]:
        """
        Returns:
            Dictionary of Assymetric Laplace distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }


class GNV2(Infinite):
    """
    This class contains methods concerning to Generalized Normal Distribution V2 [#]_. 

    Args:
        loc (float): location parameter :math:`\\mean`
        scale (float): scale parameter :math:`\\alpha`
        shape (float): shape parameter :math:`\\beta`
        x (float): random variable
    """

    def __init__(self, loc: float, scale: float, shape: float) -> None:
        self.loc = loc
        self.scale = scale
        self.shape = shape

    def pdf(self, x: Union[List[float], np.ndarray, float]
            ) -> Union[float, np.ndarray]: ...

    def cdf(self, x: Union[List[float], np.ndarray, float]
            ) -> Union[float, np.ndarray]: ...

    def mean(self) -> float: ...

    def median(self) -> float: ...

    def mode(self) -> float: ...

    def var(self) -> float: ...

    def std(self) -> float: ...

    def skewness(self) -> float: ...

    def kurtosis(self) -> float: ...

    def entropy(self) -> float: ...

    def summary(self) -> Dict[str, float]:
        """
        Returns:
            Dictionary of Assymetric Laplace distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }


class GH(Infinite):
    """
    This class contains methods concerning to Generalized Hyperbolic Distribution V1 [#]_. 

    Args:
        alpha (float): alpha parameter :math:`\\alpha`
        lmbda (float): lambda parameter :math:`\\lambda`
        asym (float): asymmetry parameter :math:`\\beta`
        scale (float): scale parameter :math:`\\delta`
        loc (float): location parameter :math:`\\mu`
        x (float): random variable
    """

    def __init__(self, lmbda: float, alpha: float, asym: float, scale: float, loc: float) -> None:
        self.lmbda = lmbda
        self.alpha = alpha
        self.asym = asym
        self.scale = scale
        self.loc = loc
        self.gamma = m.sqrt(alpha**2 - asym**2)

    def pdf(self, x: Union[List[float], np.ndarray, float]
            ) -> Union[float, np.ndarray]: ...

    def cdf(self, x: Union[List[float], np.ndarray, float]
            ) -> Union[float, np.ndarray]: ...

    def mean(self) -> float: ...

    def var(self) -> float: ...

    def std(self) -> float: ...

    def summary(self) -> Dict[str, float]:
        """
        Returns:
            Dictionary of Assymetric Laplace distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }


class HyperbolicSecant(Infinite):
    """
    This class contains methods concerning to Hyperbolic Secant [#]_. 

    .. math::
        \\text{HyperbolicSecant}(x) = \\frac{1}{2} \\sech \\Big(\\frac{\\pi}{2} x \\Big)

    Args:
        x (float): random variable

    Referneces: 
        .. [#] Wikipedia Contributors (2020). Hyperbolic secant Distirbution. https://en.wikipedia.org/wiki/Hyperbolic_secant_distribution.
    """

    def pdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]:
        if isinstance(x, (List, np.ndarray)):
            if not type(x) is np.ndarray:
                x = np.array(x)
            return 0.5*(1/np.cosh(m.pi/2*x))

        return 0.5*(1/m.cosh(m.pi/2*x))

    def cdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]:
        if isinstance(x, (List, np.ndarray)):
            if not type(x) is np.ndarray:
                x = np.array(x)
            return 2/m.pi*np.arctanh(np.exp(m.pi/2*x))

        return 2/m.pi*m.atanh(m.exp(m.pi/2*x))

    def mean(self) -> float:
        return 0.0

    def median(self) -> float:
        return 0.0

    def mode(self) -> float:
        return 0.0

    def var(self) -> float:
        return 1.0

    def std(self) -> float:
        return 1.0

    def skewness(self) -> float:
        return 0.0

    def kurtosis(self) -> float:
        return 2.0

    def entropy(self) -> float:
        # 4/pi*CatalanConstant
        return 1.16624

    def summary(self) -> Dict[str, float]:
        """
        Returns:
            Dictionary of Assymetric Laplace distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }


class Slash(Infinite):
    """
    This class contains methods concerning to Slash Distribution [#]_. 

    .. math::
        \\text{Slash}(x) = {\\displaystyle {\\begin{cases}{\\frac {\\varphi (0)-\\varphi (x)}{x^{2}}}&x\\neq 0\\{\\frac {1}{2{\\sqrt {2\\pi }}}}&x=0\\ \\end{cases}}}

    Args:
        x (float): random variable

    Reference:
        .. [#] Wikipedia Contributors (2021). Slash Distribution. https://en.wikipedia.org/wiki/Slash_distribution
    """

    def __init__(self, loc: float, scale: float, shape: float) -> None:
        self.loc = loc
        self.scale = scale
        self.shape = shape

    def pdf(self, x: Union[List[float], np.ndarray, float]
            ) -> Union[float, np.ndarray]: ...

    def cdf(self, x: Union[List[float], np.ndarray, float]
            ) -> Union[float, np.ndarray]: ...

    def mean(self) -> str:
        return 'Does not Exist'

    def median(self) -> float:
        return 0.0

    def mode(self) -> float:
        return 0.0

    def var(self) -> str:
        return 'Does not Exist'

    def std(self) -> str:
        return 'Does not Exist'

    def skewness(self) -> str:
        return 'Does not Exist'

    def kurtosis(self) -> str:
        return 'Does not Exist'

    def entropy(self) -> float: ...

    def summary(self) -> Dict[str, Union[float, str]]:
        """
        Returns:
            Dictionary of Assymetric Laplace distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }


class SkewNormal(Infinite):
    """
    This class contains methods concerning to Generalized Normal Distribution V1 [#]_. 

    .. math::
        \\text{SkewNormal}(x;\\xi,\\omega,\\alpha) = \\frac{2}{\\omega \\sqrt{2\\pi}} e^{-\\frac{(x-\\xi)^2}{2\\omega^2}} \\int_{-\\infty}^{1\\Big(\\frac{x-\\xi}{\\omega}\\Big)} \\frac{1}{\\sqrt{2\\pi}} e^{-\\frac{t^2}{2}} \\ dt

    Args:
        loc (float): location parameter :math:`\\xi`
        scale (float): scale parameter :math:`\\omega`
        shape (float): shape parameter :math:`\\alpha`
        x (float): random variable

    Reference: 
        .. [#] Wikipedia Contributors (2021). Skew Normal Distribution. https://en.wikipedia.org/wiki/Skew_normal_distribution.
    """

    def __init__(self, loc: float, scale: float, shape: float) -> None:
        self.loc = loc
        self.scale = scale
        self.shape = shape

    def pdf(self, x: Union[List[float], np.ndarray, float]
            ) -> Union[float, np.ndarray]: ...

    def cdf(self, x: Union[List[float], np.ndarray, float]
            ) -> Union[float, np.ndarray]: ...

    def mean(self) -> float: ...

    def median(self) -> float: ...

    def mode(self) -> float: ...

    def var(self) -> float: ...

    def std(self) -> float: ...

    def skewness(self) -> float: ...

    def kurtosis(self) -> float: ...

    def summary(self) -> Dict[str, Optional[float]]:
        """
        Returns:
            Dictionary of Assymetric Laplace distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }


class Landau(Infinite):
    """
    This class contains methods concerning to Generalized Normal Distribution V1 [#]_. 

    Args:
        scale (float): scale parameter :math:`\\c > 0`
        loc (float): location parameter :math:`\\mu`
        x (float): random variable
    """

    def __init__(self, scale: float, loc: float) -> None:
        self.scale = scale
        self.loc = loc

    def pdf(self, x: Union[List[float], np.ndarray, float]
            ) -> Union[float, np.ndarray]: ...

    def cdf(self):
        return NotImplemented

    def mean(self) -> str:
        return 'Undefined'

    def var(self) -> str:
        return 'Undefined'

    def std(self) -> str:
        return 'Undefined'

    def summary(self) -> Dict[str, str]:
        """
        Returns:
            Dictionary of Assymetric Laplace distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }


class JohnsonSU(Infinite):
    """
    This class contains methods concerning to Generalized Normal Distribution V1 [#]_. 


    .. math::
        \\text{JohnsonSU}(x;\\gamma,\\xi,\\delta,\\lambda) = \\frac{\\delta}{\\lambda \\sqrt{2\\pi} \\cdot \\sqrt{1+\\Big(\\frac{x-\\xi}{\\lambda}\\Big)}} e^{-0.5\\Big(\\gamma + \\delta \\sinh^-1 \\Big(\\frac{x-\\xi}{\\lambda}\\Big) \\Big)^2}

    Args:
        gamma (float): gamma parameter :math:`\\gamma`
        xi (float): xi parameter :math:`\\xi`
        delta (float): delta parameter :math:`\\delta > 0`
        lamlda (float): lambda parameter :math:`\\lambda > 0`
        x (float): random variable

    Reference:
        .. [#] Wikipedia Contributors (2021). Johnson's SU-distribution. https://en.wikipedia.org/wiki/Johnson.
    """

    def __init__(self, gamma: float, xi: float, delta: float, lmbda: float) -> None:
        if delta <= 0 or lmbda <= 0:
            raise ValueError(
                'delta and lmbda parameter are expected to be greater than 0')

        self.gamma = gamma
        self.xi = xi
        self.delta = delta
        self.lmbda = lmbda

    def pdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]:

        gamma, xi = self.gamma, self.xi
        delta, lmbda = self.delta, self.lmbda

        x0 = delta/(lmbda*m.sqrt(m.pi*2))

        if isinstance(x, (List, np.ndarray)):
            if not type(x) is np.ndarray:
                x = np.array(x)
            x1 = 1/np.sqrt(1+np.power((x-xi)/lmbda, 2))
            return x0*x1*np.exp(-0.5*np.power(gamma + delta*np.arcsinh((x-xi)/lmbda), 2))

        x1 = 1/m.sqrt(1+pow((x-xi)/lmbda, 2))
        return x0*x1*m.exp(-0.5*pow(gamma + delta*m.asinh((x-xi)/lmbda), 2))

    def cdf(self, x: Union[List[float], np.ndarray, float]
            ) -> Union[float, np.ndarray]: ...

    def mean(self) -> float:
        delta = self.delta
        return self.xi - self.lmbda * m.exp(pow(delta, -2)/2)*m.sinh(self.gamma/delta)

    def median(self) -> float:
        return self.xi + self.lmbda*m.sinh(-self.gamma/self.delta)

    def var(self) -> float:
        delta = self.delta
        x0 = pow(self.lmbda, 2)/2*(m.exp(delta**-2) - 1)
        return x0*m.exp(pow(delta, -2)*m.cosh(2*self.gamma/delta) + 1)

    def std(self) -> float:
        return m.sqrt(self.var())

    def summary(self) -> Dict[str, Optional[float]]:
        """
        Returns:
            Dictionary of Assymetric Laplace distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }


class VarianceGamma(Infinite):
    """
    This class contains methods concerning to Generalized Normal Distribution V1 [#]_. 

    Args:
        loc (float): location parameter :math:`\\mean`
        scale (float): scale parameter :math:`\\alpha`
        shape (float): shape parameter :math:`\\beta`
        x (float): random variable
    """

    def __init__(self, loc: float, scale: float, shape: float) -> None:
        self.loc = loc
        self.scale = scale
        self.shape = shape

    def pdf(self, x: Union[List[float], np.ndarray, float]
            ) -> Union[float, np.ndarray]: ...

    def cdf(self, x: Union[List[float], np.ndarray, float]
            ) -> Union[float, np.ndarray]: ...

    def mean(self) -> float: ...

    def median(self) -> float: ...

    def mode(self) -> float: ...

    def var(self) -> float: ...

    def std(self) -> float: ...

    def skewness(self) -> float: ...

    def kurtosis(self) -> float: ...

    def entropy(self) -> float: ...

    def summary(self) -> Dict[str, Optional[float]]:
        """
        Returns:
            Dictionary of Assymetric Laplace distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }
