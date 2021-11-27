try:
    import numpy as np
    import scipy.special as ss
    from typing import Union, Optional, Dict, List
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

    def __init__(self, loc: float, scale: float):
        if scale < 0:
            raise ValueError('scale should be a positive number.')
        self.scale = scale
        self.loc = loc

    def pdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]:
        """
        .. image:: ../docs/img/Infinite/CauchyPDF.png
            :width: 500

        Args:
            x (Union[List[float], numpy.ndarray, float]): random variable(s)

        Returns:
            Union[float, numpy.ndarray]: evaluation of pdf at x
        """
        loc = self.loc
        scale = self.scale

        if isinstance(x, (np.ndarray, List)):
            if not type(x) is np.ndarray:
                x = np.array(x, dtype=np.float64) 
            return 1/(m.pi * scale * (1 + np.power((x - loc) / scale, 2)))

        return 1/(m.pi * scale * (1 + pow((x - loc) / scale, 2)))

    def cdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]:
        """
        .. image:: ../docs/img/Infinite/CauchyCDF.png
            :width: 500

        Args:
            x (Union[List[float], numpy.ndarray, float]): data point(s) of interest

        Returns:
            Union[float, numpy.ndarray]: evaluation of cdf at x
        """
        loc = self.loc
        scale = self.scale

        if isinstance(x, (np.ndarray, List)):
            if not type(x) is np.ndarray:
                x = np.array(x, dtype=np.float64) 
            return (1 / m.pi) * np.arctan((x - loc) / scale) + 0.5

        return (1 / m.pi) * m.atan((x - loc) / scale) + 0.5

    def mean(self) -> str:
        """
        Returns: Mean of the Cauchy distribution. Mean is Undefined.
        """
        return "Indeterminate"

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
        return "Indeterminate"

    def std(self) -> str:
        """
        Returns: Standard Deviation of the Cauchy Distribution.
        """
        return "Indeterminate"

    def skewness(self) -> str:
        """
        Returns: Skewness of the Cauchy distribution.
        """
        return "Indeterminate"

    def kurtosis(self) -> float:
        """
        Returns: Kurtosis of the Cauchy distribution
        """
        return m.log(4 * m.pi * self.scale)

    def entropy(self) -> float:
        """
        Returns: differential entropy of the Cauchy distribution

        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier.
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        return m.log10(4*m.pi*self.scale)

    def summary(self) -> Dict[str, Union[float, str]]:
        """
        Returns:
            Dictionary of Cauchy distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }


class T(Infinite):
    """
    This class contains implementation of the Student's Distribution for calculating the
    probablity density function and cumulative distribution function. Additionally,
    a t-table __generator is also provided by p-value method. Note that the implementation
    of T(Student's) distribution is defined by beta-functions [#]_ [#]_.

    .. math::
        \\text{T}(x;\\nu) = \\frac{1}{\\sqrt{\\nu}\\text{B}\\Big(\\frac{1}{2}, \\frac{\\nu}{2}\\Big)} \\Big(1 + \\frac{t^2}{\\nu}\\Big) ^{-\\frac{\\nu+1}{2}}

    .. math::
        \\text{T}(x;\\nu) = \\frac{\\Gamma\\Big( \\frac{\\nu+1}{2} \\Big)}{\\sqrt{\\nu\\pi} \\Gamma{\\Big( \\frac{\\nu}{2} \\Big)}} \\Big(1 + \\frac{x^2}{\\nu}\\Big) ^{- \\frac{v+1}{2}} \\\\

    Args:
        df(int): degrees of freedom (:math:`\\nu`) where df > 0
        x(float): random variable 

    References:

        .. [#] Kruschke JK (2015). Doing Bayesian Data Analysis (2nd ed.). Academic Press. ISBN 9780124058880. OCLC 959632184.
        .. [#] Weisstein, Eric W. "Student's t-Distribution." From MathWorld--A Wolfram Web Resource. https://mathworld.wolfram.com/Studentst-Distribution.html

    """

    def __init__(self, df: int):
        if type(df) is not int:
            raise TypeError('degrees of freedom(df) should be a whole number.')
        if df < 0:
            raise ValueError('df parameter must not be less than 0')

        self.df = df

    def pdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]:
        """
        .. image:: ../docs/img/Infinite/TPDF.png
            :width: 500

        Args:
            x (Union[List[float], numpy.ndarray, float]): random variable(s)

        Returns:
            Union[float, numpy.ndarray]: evaluation of pdf at x
        """
        df = self.df

        if isinstance(x, (np.ndarray, List)):
            if not type(x) is np.ndarray:
                x = np.array(x, dtype=np.float64) 
            return (1 / (m.sqrt(df) * ss.beta(0.5, df / 2))) * np.power((1 + np.power(x, 2) / df), -(df + 1) / 2)

        return (1 / (m.sqrt(df) * ss.beta(0.5, df / 2))) * pow((1 + pow(x, 2) / df), -(df + 1) / 2)

    def cdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]:
        """
        .. image:: ../docs/img/Infinite/TCDF.png
            :width: 500

        Args:
            x (Union[List[float], numpy.ndarray, float]): data point(s) of interest

        Returns:
            Union[float, numpy.ndarray]: evaluation of cdf at x
        """
        df = self.df
        f1 = lambda x: 0.5*ss.betainc(df/2, 0.5, df/(x**2 + df))
        f2 = lambda x: 0.5*(ss.betainc(0.5, df/2, pow(x,2)/(x**2 + df)) + 1)

        if isinstance(x, (np.ndarray, List)):
            if not type(x) is np.ndarray:
                x = np.array(x, dtype=np.float64) 
            return np.piecewise(x, [x <= 0, x > 0], [f1, f2])

        return f1(x) if x <= 0 else f2(x)

    def mean(self) -> Union[float, str]:
        """
        Mean of the T-distribution.
        Returns:
            0 for df > 1, otherwise Indeterminate.
        """
        df = self.df
        if df > 1:
            return 0.0
        return "Indeterminate"

    def median(self) -> float:
        """
        Returns: Median of the T-distribution
        """
        return 0.0

    def mode(self) -> float:
        """
        Returns: Mode of the T-distribution
        """
        return 0.0

    def var(self) -> Union[float, str]:
        """
        Returns: Variance of the T-distribution
        """
        df = self.df
        if df > 2:
            return df / (df - 2)
        if df > 1 and df <= 2:
            return np.inf
        return "Indeterminate"

    def std(self) -> Union[float, str]:
        """
        Returns: Standard Deviation of the T-distribution
        """
        var = self.var()
        if type(var) is float:
            return m.sqrt(var)
        return "Indeterminate"

    def skewness(self) -> Union[float, str]:
        """
        Returns: Skewness of the T-distribution
        """
        df = self.df
        if df > 3:
            return 0.0
        return "Indeterminate"

    def kurtosis(self) -> Union[float, str]:
        """
        Returns: Kurtosis of the T-distribution
        """
        df = self.df
        if df > 4:
            return 6 / (df - 4)
        if df > 2 and df <= 4:
            return float('inf')
        return "Indeterminate"

    def entropy(self) -> float:
        """
        Returns: differential entropy of T-distribution

        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier.
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        df = self.df
        return (df+1)/2 * (ss.digamma((df+1)/2)-ss.digamma(df/2)) + m.log(m.sqrt(df)*ss.beta(df/2, 1/2))

    def summary(self) -> Dict[str, Union[float, str]]:
        """
        Returns:
            Dictionary of T distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }


class Gaussian(Infinite):
    """
    This class contains methods concerning the Gaussian Distribution [#]_ [#]_.

    .. math::
        \\text{Gaussian}(x;\\mu,\\sigma) = \\frac{1}{\\sigma \\sqrt{2 \\pi}} e^{-\\frac{1}{2}\\big( \\frac{x-\\mu}{\\sigma}\\big)^2}

    Args:

        mean(float): mean of the distribution (:math:`\\mu`)
        std(float): standard deviation (:math:`\\sigma`) of the distribution where std > 0
        x(float): random variable 

    References:
        .. [#] Wikipedia contributors. (2020, December 19). Gaussian distribution. https://en.wikipedia.org/w/index.php?title=Gaussian_distribution&oldid=995237372
        .. [#] Weisstein, Eric W. "Gaussian Distribution." From MathWorld--A Wolfram Web Resource. https://mathworld.wolfram.com/GaussianDistribution.html

    """

    def __init__(self, mean: float = 0, stdev: float = 1):
        if stdev < 0:
            raise ValueError("stdev parameter must not be less than 0.")

        self.mean_val = mean
        self.stdev = stdev

    def pdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]:
        """
        .. image:: ../docs/img/Infinite/GaussianPDF.png
            :width: 500

        Args:
            x (Union[List[float], numpy.ndarray, float]): random variable(s)

        Returns:
            Union[float, numpy.ndarray]: evaluation of pdf at x
        """
        mean = self.mean_val
        std = self.stdev
        c0 = 2.5066282746310002  # m.sqrt(2*m.pi)
        x0 = std*c0

        if isinstance(x, (np.ndarray, List)):
            if not type(x) is np.ndarray:
                x = np.array(x, dtype=np.float64) 
            return np.exp(-0.5*np.power((x-mean)/std, 2))/x0

        return m.exp(-0.5*pow((x-mean)/std, 2))/x0

    def cdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]:
        """
        .. image:: ../docs/img/Infinite/GaussianCDF.png
            :width: 500

        Args:
            x (Union[List[float], numpy.ndarray, float]): data point(s) of interest

        Returns:
            Union[float, numpy.ndarray]: evaluation of cdf at x
        """

        if isinstance(x, (np.ndarray, List)):
            if not type(x) is np.ndarray:
                x = np.array(x, dtype=np.float64) 
        return 0.5*(1+ss.erf((x-self.mean_val)/(self.stdev*m.sqrt(2))))

    def mean(self) -> float:
        """
        Returns: Mean of the Gaussian distribution
        """
        return self.mean_val

    def median(self) -> float:
        """
        Returns: Median of the Gaussian distribution
        """
        return self.mean_val

    def mode(self) -> float:
        """
        Returns: Mode of the Gaussian distribution
        """
        return self.mean_val

    def var(self) -> float:
        """
        Returns: Variance of the Gaussian distribution
        """
        return pow(self.stdev, 2)

    def std(self) -> float:
        """
        Returns: Standard deviation of the Gaussian distribution
        """
        return self.stdev

    def skewness(self) -> float:
        """
        Returns: Skewness of the Gaussian distribution
        """
        return 0.0

    def kurtosis(self) -> float:
        """
        Returns: Kurtosis of the Gaussian distribution
        """
        return 0.0

    def entropy(self) -> float:
        """
        Returns: differential entropy of the Gaussian distribution

        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier.
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        return m.log(self.std()*m.sqrt(2 * m.pi * m.e))

    def summary(self) -> Dict[str, Union[float, int, str]]:
        """
        Returns:
            Dictionary of Gaussian distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }


class Laplace(Infinite):
    """
    This class contains methods concerning Laplace Distirbution [#]_ [#]_.

    .. math::
        \\text{Laplace}(x;\\mu, b) = \\frac{1}{2b} \\exp{- \\frac{|x - \\mu |}{b}}

    Args:

        loc(float): loc parameter (:math:`\\mu`)
        scale(float): scale parameter (:math:`b > 0`) 
        x(float): random variable

    Reference:
        .. [#] Wikipedia contributors. (2020, December 21). Laplace distribution. https://en.wikipedia.org/w/index.php?title=Laplace_distribution&oldid=995563221
        .. [#] Wolfram Research (2007), LaplaceDistribution, Wolfram Language function, https://reference.wolfram.com/language/ref/LaplaceDistribution.html (updated 2016).
    """

    def __init__(self, loc: float, scale: float):
        if scale < 0:
            raise ValueError('scale should be greater than 0.')

        self.scale = scale
        self.loc = loc

    def pdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]:
        """
        .. image:: ../docs/img/Infinite/LaplacePDF.png
            :width: 500

        Args:
            x (Union[List[float], numpy.ndarray, float]): random variable(s)

        Returns:
            Union[float, numpy.ndarray]: evaluation of pdf at x
        """
        mu = self.loc
        b = self.scale

        if isinstance(x, (np.ndarray, List)):
            if not type(x) is np.ndarray:
                x = np.array(x, dtype=np.float64) 
            return (1 / (2 * b)) * np.exp(- np.abs(x - mu) / b)
        return (1 / (2 * b)) * m.exp(- abs(x - mu) / b)

    def cdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]:
        """
        .. image:: ../docs/img/Infinite/LaplaceCDF.png
            :width: 500

        Args:
            x (Union[List[float], numpy.ndarray, float]): data point(s) of interest

        Returns:
            Union[float, numpy.ndarray]: evaluation of cdf at x
        """
        mu = self.loc
        b = self.scale

        f0 = lambda x: 1 - 0.5*np.exp(-(x-mu)/b)
        f1 = lambda x: 0.5*np.exp((x-mu)/b)

        if isinstance(x, (np.ndarray, List)):
            if not type(x) is np.ndarray:
                x = np.array(x, dtype=np.float64) 
            return np.piecewise(x, [x >= mu, x < mu], [f0,f1])

        return f0(x) if x >= mu else f1(x)

    def mean(self) -> float:
        """
        Returns: Mean of the Laplace distribution.
        """
        return self.loc

    def median(self) -> float:
        """
        Returns: Median of the Laplace distribution.
        """
        return self.loc

    def mode(self) -> float:
        """
        Returns: Mode of the Laplace distribution.
        """
        return self.loc

    def var(self) -> Union[int, float]:
        """
        Returns: Variance of the Laplace distribution.
        """
        return 2 * pow(self.scale, 2)

    def std(self) -> float:
        """
        Returns: Standard deviation of the Laplace distribution
        """
        return m.sqrt(self.var())

    def skewness(self) -> float:
        """
        Returns: Skewness of the Laplace distribution.
        """
        return 0.0

    def kurtosis(self) -> float:
        """
        Returns: Kurtosis of the Laplace distribution.
        """
        return 3.0

    def entropy(self) -> float:
        """
        Returns: differential entropy of the Laplace distribution.

        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier.
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        return 1 + m.log(2*self.scale)

    def summary(self) -> Dict[str, Union[float, str]]:
        """
        Returns:
            Dictionary of Laplace distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }


class Logistic(Infinite):
    """
    This class contains methods concerning Logistic Distirbution [#]_ [#]_.

    .. math::
        \\text{Logistic}(x;\\mu,s) = \\frac{\\exp{(-(x-\\mu)/s)}} {s(1+\\exp(-(x-\\mu)/s)^2)}

    Args:

        location(float): location parameter (:math:`\\mu`)
        scale(float): scale parameter (:math:`s`) x > 0 
        x(float): random variable

    Reference:
        .. [#] Wikipedia contributors. (2020, December 12). Logistic distribution. https://en.wikipedia.org/w/index.php?title=Logistic_distribution&oldid=993793195
        .. [#] Wolfram Research (2007), LogisticDistribution, Wolfram Language function, https://reference.wolfram.com/language/ref/LogisticDistribution.html (updated 2016).
    """

    def __init__(self, location: float, scale: float):
        if scale < 0:
            raise ValueError('scale should be greater than 0.')

        self.scale = scale
        self.location = location

    def pdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]:
        """
        .. image:: ../docs/img/Infinite/LogisticPDF.png
            :width: 500

        Args:
            x (Union[List[float], numpy.ndarray, float]): random variable(s)

        Returns:
            Union[float, numpy.ndarray]: evaluation of pdf at x
        """
        mu = self.location
        s = self.scale

        if isinstance(x, (np.ndarray, List)):
            if not type(x) is np.ndarray:
                x = np.array(x, dtype=np.float64) 
            return np.exp(-(x - mu) / s) / (s * (1 + np.exp(-(x - mu) / s))**2)
        return m.exp(-(x - mu) / s) / (s * (1 + m.exp(-(x - mu) / s))**2)

    def cdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]:
        """
        .. image:: ../docs/img/Infinite/LogisticCDF.png
            :width: 500

        Args:
            x (Union[List[float], numpy.ndarray, float]): data point(s) of interest

        Returns:
            Union[float, numpy.ndarray]: evaluation of cdf at x
        """
        mu = self.location
        s = self.scale

        if isinstance(x, (np.ndarray, List)):
            x = np.array(x, dtype=np.float64) 
            return 1 / (1 + np.exp(-(x - mu) / s))
        return 1 / (1 + m.exp(-(x - mu) / s))

    def mean(self) -> float:
        """
        Returns: Mean of the Logistic distribution.
        """
        return self.location

    def median(self) -> float:
        """
        Returns: Median of the Logistic distribution.
        """
        return self.location

    def mode(self) -> float:
        """
        Returns: Mode of the Logistic distribution.
        """
        return self.location

    def var(self) -> float:
        """
        Returns: Variance of the Logistic distribution.
        """
        return pow(self.scale, 2) * pow(m.pi, 2)/3

    def std(self) -> float:
        """
        Returns: Standard deviation of the Logistic distribution.
        """
        return m.sqrt(self.var())

    def skewness(self) -> float:
        """
        Returns: Skewness of the Logistic distribution.
        """
        return 0.0

    def kurtosis(self) -> float:
        """
        Returns: Kurtosis of the Logistic distribution.
        """
        return 6 / 5

    def entropy(self) -> float:
        """
        Returns: differential entropy of the Logistic distribution.

        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier.
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        return 2.0

    def summary(self) -> Dict[str, Union[float, str]]:
        """
        Returns:
            Dictionary of Logistic distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }

# Todo: Implement moments of Fisher-Z Distribution. 
class FisherZ(Infinite):
    """
    This class contains methods concerning Fisher's z-Distribution [#]_ [#]_ [#]_.

    .. math:: 
        \\text{Fisher}(x;d_1, d_2) = \\frac{2d_2^{d_1/2} d_2^{d_2/2}}{\\text{B}\\Big(\\frac{d_1}{2}, \\frac{d_2}{2}\\Big)} \\frac{e^{d_1 x}}{(d_1 e^{2x} + d_2)^{(d_1+d_2)/2}}

    Args:

        df1(float): degrees of freedom (:math:`d_1 > 0`).
        df2(float): degrees of freedom (:math:`d_2 > 0`).
        x(float): random variable.

    Note: Fisher's z-distribution is the statistical distribution of half the log of an F-distribution variate:
    z = 1/2*log(F)

    Reference:
        .. [#] Wikipedia contributors. (2020, December 15). Fisher's z-distribution. https://en.wikipedia.org/w/index.php?title=Fisher%27s_z-distribution&oldid=994427156.
        .. [#] Wolfram Research (2010), FisherZDistribution, Wolfram Language function, https://reference.wolfram.com/language/ref/FisherZDistribution.html (updated 2016).
        .. [#] Wolfram Alpha (2021). Fisher Distribution. https://www.wolframalpha.com/input/?i=Fisher+distribution.
    """

    def __init__(self, df1: float, df2: float):
        if df1 <= 0 or df2 <= 0:
            raise ValueError('degrees of freedom  are expected to be positive')

        self.df1 = df1
        self.df2 = df2

    def pdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]:
        """
        .. image:: ../docs/img/Infinite/FisherZPDF.png
            :width: 500

        Args:
            x (Union[List[float], numpy.ndarray, float]): random variable(s)

        Returns:
            Union[float, numpy.ndarray]: evaluation of pdf at x
        """
        df1, df2 = self.df1, self.df2

        x0 = (2*pow(df1, df1/2)*pow(df2, df2/2))
        if isinstance(x, (List, np.ndarray)):
            if not type(x) is np.ndarray:
                x = np.array(x, dtype=np.float64) 
            x1 = x0*np.exp(df1*x)
            x2 = ss.beta(df1/2, df2/2) * np.power(df1 *
                                                  np.exp(2*x)+df2, (df1+df2)*0.5)
            return x1/x2

        x1 = x0*m.exp(df1*x)
        x2 = ss.beta(df1/2, df2/2) * pow(df1*m.exp(2*x)+df2, (df1+df2)*0.5)
        return x1/x2

    def cdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]:
        """
        .. image:: ../docs/img/Infinite/FisherZCDF.png
            :width: 500

        Args:
            x (Union[List[float], np.ndarray, float]): data point(s) of interest   

        Returns:
            Union[float, np.ndarray]: evaluation of cdf at x 
        """
        df1, df2 = self.df1, self.df2

        if isinstance(x, (List, np.ndarray)):
            if not type(x) is np.ndarray:
                x = np.array(x, np.float64)

        return ss.betainc(df1/2, df2/2, df1*np.exp(2*x)/(df2 + df1*np.exp(2*x)))

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
        scale (float): scale parameter :math:`\\lambda > 0`
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
        .. image:: ../docs/img/Infinite/AssymetricLaplacePDF.png
            :width: 500

        Args:
            x (Union[List[float], numpy.ndarray, float]): random variable(s)

        Returns:
            Union[float, numpy.ndarray]: evaluation of pdf at x
        """
        l, k, loc = self.scale, self.asym, self.loc

        x0 = l/(k+1/k)
        if isinstance(x, (List, np.ndarray)):
            if not type(x) is np.ndarray:
                x = np.array(x, dtype=np.float64) 

            def f01(x): return np.exp(l/k*(x-loc))
            def f02(x): return np.exp(-l*k*(x-loc))
            return x0*np.piecewise(x, [x < loc, x >= loc], [f01, f02])

        def f1(x): return m.exp(l/k*(x-loc))
        def f2(x): return m.exp(-l*k*(x-loc))

        return x0*(f1(x) if x < loc else f2(x))

    def cdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]:
        """
        .. image:: ../docs/img/Infinite/AssymetricLaplaceCDF.png
            :width: 500

        Args:
            x (Union[List[float], numpy.ndarray, float]): data point(s) of interest

        Returns:
            Union[float, numpy.ndarray]: evaluation of cdf at x
        """
        l, k, loc = self.scale, self.asym, self.loc

        if isinstance(x, (List, np.ndarray)):
            if not type(x) is np.ndarray:
                x = np.array(x, dtype=np.float64) 

            def f01(x): return k**2/(1+k**2)*np.exp(l/k*(x-loc))
            def f02(x): return 1 - 1/(1+k**2)*np.exp(-l*k*(x-loc))
            return np.piecewise(x, [x <= loc, x > loc], [f01, f02])

        def f1(x): return k**2/(1+k**2)*m.exp(l/k*(x-loc))
        def f2(x): return 1 - 1/(1+k**2)*np.exp(-l*k*(x-loc))
        return f1(x) if x <= loc else f2(x)

    def mean(self) -> float:
        """
        Returns: Mean of the Assymetric Laplace distribution.
        """
        k = self.asym

        return self.loc + (1+k**2)/(self.scale*k)

    def median(self) -> Optional[float]:
        """
        Returns: Median of the Assymetric Laplace distribution.
        """
        loc = self.loc
        k = self.asym
        l = self.scale

        if k > 1:
            return loc + k/l*m.log((1+k**2)/(2*k**2))
        if k < 1:
            return loc + 1/(l*k)*m.log((1+k**2)/2)

    def var(self) -> float:
        """
        Returns: Variance of the Assymetric Laplace distribution.
        """
        k = self.asym
        return (1+k**4)/(self.scale**2*k**2)

    def std(self) -> float:
        """
        Returns: Standard Deviation of the Assymetric Laplace distribution.
        """
        k = self.asym
        return m.sqrt((1+k**4)/(self.scale**2*k**2))

    def skewness(self) -> float:
        """
        Returns: Skewness of the Assymetric Laplace distribution.
        """
        k = self.asym
        return (2*(1-k**6))/pow(k**4+1, 1.5)

    def kurtosis(self) -> float:
        """
        Returns: Kurtosis of the Assymetric Laplace distribution.
        """
        k = self.asym
        return 6*(1+k**8)/pow(1+k**4, 2)

    def entropy(self) -> float:
        """
        Returns: entropy of the Assymetric Laplace distribution.
        """
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

    .. math:: 
        \\text{GNV1}(x; \\mu, \\alpha, \\beta) = \\frac{\\beta}{2 \\alpha \\Gamma(1/\\beta)} e^{(- |x-\\mu|/ \\alpha)^\\beta}

    Args:
        loc (float): location parameter :math:`\\mu`
        scale (float): scale parameter :math:`\\alpha`
        shape (float): shape parameter :math:`\\beta`
        x (float): random variable

    Reference:
        .. [#] Wikipedia Contributors (2021). Generalized normal distribution. https://en.wikipedia.org/wiki/Generalized_normal_distribution.
    """

    def __init__(self, loc: float, scale: float, shape: float) -> None:
        self.loc = loc
        self.scale = scale
        self.shape = shape

    def pdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]:
        """
        Args:
            x (Union[List[float], numpy.ndarray, float]): random variable(s)

        Returns:
            Union[float, numpy.ndarray]: evaluation of pdf at x
        """
        b, a, mu = self.shape, self.scale, self.loc
        x0 = b/(2*a*ss.gamma(1/b))

        if isinstance(x, (List, np.ndarray)):
            if not type(x) is np.ndarray:
                x = np.array(x, dtype=np.float64) 
            return x0*np.exp(np.power(-np.abs(x-mu)/a, b))
        return x0*m.exp(pow(-abs(x-mu)/a, b))

    def cdf(self, x: Union[List[float], np.ndarray, float]
            ) -> Union[float, np.ndarray]: ...

    def mean(self) -> float:
        """
        Returns: Mean of the GNV1 distribution.
        """
        return self.loc

    def median(self) -> float:
        """
        Returns: Median of the GNV1 distribution.
        """
        return self.loc

    def mode(self) -> float:
        """
        Returns: Mode of the GNV1 distribution.
        """
        return self.loc

    def var(self) -> float:
        """
        Returns: Variance of the GNV1 distribution.
        """
        a, b = self.scale, self.shape
        return a**2*ss.gamma(3/b)/ss.gamma(1/b)

    def std(self) -> float:
        """
        Returns: Standard Deviation of the GNV1 distribution.
        """
        a, b = self.scale, self.shape
        return m.sqrt(a**2*ss.gamma(3/b)/ss.gamma(1/b))

    def skewness(self) -> float:
        """
        Returns: Skewness of the GNV1 distribution.
        """
        return 0.0

    def kurtosis(self) -> float:
        """
        Returns: Kurtosis of the GNV1 distribution.
        """
        b = self.shape
        return ss.gamma(5/b)*ss.gamma(1/b)/ss.gamma(3/b)**2 - 3

    def entropy(self) -> float:
        """
        Returns: Entropy of the GNV1 distribution.
        """
        b = self.shape
        return 1/b - m.log(b/(2*self.scale*ss.gamma(1/b)))

    def summary(self) -> Dict[str, Optional[float]]:
        """
        Returns:
            Dictionary of GNV1 distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }


class GNV2(Infinite):
    """
    This class contains methods concerning to Generalized Normal Distribution V2 [#]_. 

    Args:
        loc (float): location parameter :math:`\\xi`
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
            Dictionary of GNV2 distirbution moments. This includes standard deviation. 
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
            Dictionary of GH distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }


class HyperbolicSecant(Infinite):
    """
    This class contains methods concerning to Hyperbolic Secant [#]_ [#]_ . 

    .. math::
        \\text{HyperbolicSecant}(x; \\mu, \\sigma) = \\frac{\\text{sech} \\Big( \\frac{\\pi (x - \\mu)}{2 \\sigma} \\Big)}{2 \\sigma}

    Args:
        loc (float): location parameter :math:`\\mu`
        scale (float): scale parameter :math:`\\sigma > 0`
        x (float): random variable

    Referneces: 
        .. [#] Seigrist, K. (n.d.) The Hyperbolic Secant Distribution. https://www.randomservices.org/random/special/HyperbolicSecant.html.
        .. [#] Wolfram Alpha(2021). Hyperbolic Secant. https://www.wolframalpha.com/input/?i=hyberbolic+secant+distribution.
        
    """

    def __init__(self, loc:float, scale:float): 
        if scale <= 0:
            raise ValueError('scale parameter is expected to be a positive real number')

        self.loc, self.scale = loc, scale

    def pdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]:
        """
        .. image:: ../docs/img/Infinite/HyperbolicSecantPDF.png
            :width: 500

        Args:
            x (Union[List[float], numpy.ndarray, float]): random variable(s)

        Returns:
            Union[float, numpy.ndarray]: evaluation of pdf at x
        """
        mu, sigma = self.loc, self.scale

        if isinstance(x, (List, np.ndarray)):
            if not type(x) is np.ndarray:
                x = np.array(x, dtype=np.float64) 
        
        x0 = 2*sigma
        x1 = 1/np.cosh(m.pi*(x-mu)/(x0))
        return x1 / x0

    def cdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]:
        """
        .. image:: ../docs/img/Infinite/HyperbolicSecantCDF.png
            :width: 500

        Args:
            x (Union[List[float], numpy.ndarray, float]): data point(s) of interest

        Returns:
            Union[float, numpy.ndarray]: evaluation of cdf at x
        """
        mu, sigma = self.loc, self.scale
        if isinstance(x, (List, np.ndarray)):
            if not type(x) is np.ndarray:
                x = np.array(x, dtype=np.float64) 

        return 2*np.arctan(np.exp(m.pi*(x - mu)/ (2*sigma)))/ m.pi

    def mean(self) -> float: 
        """
        Returns: Mean of Hyperbolic Secant Distribution.
        """
        return self.loc

    def mode(self) -> float: 
        """
        Returns: Mode of Hyperbolic Secant Distribution.
        """
        return self.loc

    def var(self) -> float: 
        """
        Returns: Variance of Hyperbolic Secant Distribution.
        """
        return self.scale**2

    def std(self) -> float: 
        """
        Returns: Standard Deviation of Hyperbolic Secant Distribution.
        """
        return self.scale

    def skewness(self) -> float: 
        """
        Returns: Skewness of Hyperbolic Secant Distribution.
        """
        return 0.0

    def kurtosis(self) -> float:
        """
        Returns: Kurtosis of Hyperbolic Secant Distribution.
        """
        return 5.0

    def summary(self) -> Dict[str, float]:
        """
        Returns:
            Dictionary of Hyperbolic Secant distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }


class Slash(Infinite):
    """
    This class contains methods concerning to Slash Distribution [#]_. 

    .. math:: 
        \text{Slash}(x) = {\displaystyle {\begin{cases}{\frac {\varphi (0)-\varphi (x)}{x^{2}}}&x\neq 0\\{\frac {1}{2{\sqrt {2\pi }}}}&x=0\\\end{cases}}}
    
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
        """
        Returns: Mean of the Slash distribution.
        """
        return 'Does not Exist'

    def median(self) -> float:
        """
        Returns: Median of the Slash distribution.
        """
        return 0.0

    def mode(self) -> float:
        """
        Returns: Mode of the Slash distribution.
        """
        return 0.0

    def var(self) -> str:
        """
        Returns: Variance of the Slash distribution.
        """
        return 'Does not Exist'

    def std(self) -> str:
        """
        Returns: Standard Deviation of the Slash distribution.
        """
        return 'Does not Exist'

    def skewness(self) -> str:
        """
        Returns: Skewness of the Slash distribution.
        """
        return 'Does not Exist'

    def kurtosis(self) -> str:
        """
        Returns: Kurtosis of the Slash distribution.
        """
        return 'Does not Exist'

    def entropy(self) -> float: ...

    def summary(self) -> Dict[str, Union[float, str]]:
        """
        Returns:
            Dictionary of Slash distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }


class SkewNormal(Infinite):
    """
    This class contains methods concerning to Generalized Normal Distribution V1 [#]_ [#]_. 

    .. math::
        \\text{SkewNormal}(x;\\xi,\\omega,\\alpha) = \\frac{e^{\\frac{-(x-\\xi)^2}{2 \\omega^2} \\text{erfc} \\Big( - \\frac{a(x-\\xi)}{\\sqrt{2} \\omega} \\Big) }}{ \\sqrt{2\\pi} \\omega}

    Args:
        loc (float): location parameter :math:`\\xi`
        scale (float): scale parameter :math:`\\omega`
        shape (float): shape parameter :math:`\\alpha`
        x (float): random variable

    Reference: 
        .. [#] Wikipedia Contributors (2021). Skew Normal Distribution. https://en.wikipedia.org/wiki/Skew_normal_distribution.
        .. [#] Wolfram Research (2010), SkewNormalDistribution, Wolfram Language function, https://reference.wolfram.com/language/ref/SkewNormalDistribution.html (updated 2016).

    """

    def __init__(self, loc: float, scale: float, shape: float) -> None:
        if scale <= 0:
            raise ValueError(
                'scale parameter is expected to be a positive real number')

        self.loc = loc
        self.scale = scale
        self.shape = shape

    def pdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]:
        """
        .. image:: ../docs/img/Infinite/SkewNormalPDF.png
            :width: 500

        Args:
            x (Union[List[float], numpy.ndarray, float]): random variables

        Returns:
            Union[float, numpy.ndarray]: evaluation of pdf at x
        """
        loc, scale, shape = self.loc, self.scale, self.shape
        # x0 = m.sqrt(2*m.pi)*scale

        if isinstance(x, (List, np.ndarray)):
            if not type(x) is np.ndarray:
                x = np.array(x, dtype=np.float64)
        return np.exp(-(x-loc)**2/(2*scale**2))*ss.erfc(-(shape*(x-loc)/(m.sqrt(2)*scale)))

    def cdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]:
        """
        .. image:: ../docs/img/Infinite/SkewNormalPDF.png
            :width: 500

        Args:
            x (Union[List[float], numpy.ndarray, float]): data point(s) of interest

        Returns:
            Union[float, numpy.ndarray]: evaluation of pdf at x
        """        
        loc, scale, shape = self.loc, self.scale, self.shape

        if isinstance(x, (List, np.ndarray)):
            if not type(x) is np.ndarray:
                x = np.array(x, dtype=np.float64)
        return 0.5*ss.erfc(-(x-loc)/(m.sqrt(2)*scale)) - 2*ss.owens_t((x-loc)/scale, shape)

    def mean(self) -> float:
        """
        Returns: Mean of the Skew Normal distribution.
        """
        a, o, u = self.shape, self.scale, self.loc
        x0 = m.sqrt(2/m.pi)*a*o
        return x0/(m.sqrt(a**2 + 1)) + u

    def median(self) -> float: ...

    def mode(self) -> float: ...

    def var(self) -> float:
        """
        Returns: Variance of the Skew Normal distribution.
        """
        shape, scale = self.shape, self.scale
        x0 = 1 - 2*shape**2/(m.pi*(shape**2+1))
        return x0*pow(scale, 2)

    def std(self) -> float:
        """
        Returns: Standard Deviation of the Skew Normal distribution.
        """
        shape, scale = self.shape, self.scale
        x0 = 1 - 2*shape**2/(m.pi*(shape**2+1))
        return m.sqrt(x0*pow(scale, 2))

    def skewness(self) -> float:
        """
        Returns: Skewness of the Skew Normal distribution.
        """
        shape = self.shape
        x0 = m.sqrt(2)*(4-m.pi)*pow(shape, 3)
        return x0/pow((m.pi-2)*shape**2 + m.pi, 3/2)

    def kurtosis(self) -> float:
        """
        Returns: Kurtosis of the Skew Normal distribution.
        """
        shape = self.shape
        x0 = 8*(m.pi-3)*pow(shape, 4)
        return x0/pow((m.pi-2)*shape**2 + m.pi, 2) + 3

    def summary(self) -> Dict[str, Optional[float]]:
        """
        Returns:
            Dictionary of Assymetric Skew Normal distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }

# todo: PDF and CDF implementation
class Landau(Infinite):
    """
    This class contains methods concerning to Generalized Normal Distribution V1 [#]_. 

    Args:
        scale (float): scale parameter :math:`c > 0`
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
        """
        Returns: Variance of the Landau distribution.
        """
        return 'Indeterminate'

    def var(self) -> str:
        """
        Returns: Variance of the Landau distribution.
        """
        return 'Indeterminate'

    def std(self) -> str:
        """
        Returns: Variance of the Landau distribution.
        """
        return 'Indeterminate'

    def skewness(self) -> str:
        """
        Returns: Variance of the Landau distribution.
        """
        return 'Indeterminate'
    
    def kurtosis(self) -> str:
        """
        Returns: Variance of the Landau distribution.
        """
        return 'Indeterminate'

    def summary(self) -> Dict[str, str]:
        """
        Returns:
            Dictionary of Assymetric Landau distirbution moments. This includes standard deviation. 
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
        """
        Args:
            x (Union[List[float], numpy.ndarray, float]): random variable(s)

        Returns:
            Union[float, numpy.ndarray]: evaluation of pdf at x
        """

        gamma, xi = self.gamma, self.xi
        delta, lmbda = self.delta, self.lmbda

        x0 = delta/(lmbda*m.sqrt(m.pi*2))

        if isinstance(x, (List, np.ndarray)):
            if not type(x) is np.ndarray:
                x = np.array(x, dtype=np.float64) 
            x1 = 1/np.sqrt(1+np.power((x-xi)/lmbda, 2))
            return x0*x1*np.exp(-0.5*np.power(gamma + delta*np.arcsinh((x-xi)/lmbda), 2))

        x1 = 1/m.sqrt(1+pow((x-xi)/lmbda, 2))
        return x0*x1*m.exp(-0.5*pow(gamma + delta*m.asinh((x-xi)/lmbda), 2))

    def cdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]: 
        if isinstance(x, (List, np.ndarray)):
            x = np.array(x, dtype=np.float64) 

    def mean(self) -> float:
        """
        Returns: Variance of the JohnsonSU distribution.
        """
        delta = self.delta
        return self.xi - self.lmbda * m.exp(pow(delta, -2)/2)*m.sinh(self.gamma/delta)

    def median(self) -> float:
        """
        Returns: Variance of the JohnsonSU distribution.
        """
        return self.xi + self.lmbda*m.sinh(-self.gamma/self.delta)

    def var(self) -> float:
        """
        Returns: Variance of the JohnsonSU distribution.
        """
        delta = self.delta
        x0 = pow(self.lmbda, 2)/2*(m.exp(delta**-2) - 1)
        return x0*m.exp(pow(delta, -2)*m.cosh(2*self.gamma/delta) + 1)

    def std(self) -> float:
        """
        Returns: Variance of the JohnsonSU distribution.
        """
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
        loc (float): location parameter :math:`\\mu`
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
