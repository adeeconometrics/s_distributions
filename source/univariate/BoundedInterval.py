try:
    import numpy as np
    import scipy.special as ss
    from typing import Union, Tuple, Dict, List
    import math as m
    from univariate._base import Base
except Exception as e:
    print(f"some modules are missing {e}")

class BoundedInterval(Base):
    """
    Description:
        Base class for probability tags.
    """

    def __init__(self):
        if type(self) is BoundedInterval:
            raise TypeError('base class cannot be instantiated.')


class Arcsine(BoundedInterval):
    """
    This class contains methods concerning Arcsine Distirbution [#]_.

    .. math::
        \\text{Arcsine}(x)={\\frac{1}{\\pi \\sqrt{x(1-x)}}}

    Args:

        x(float): random variable between 0 and 1

    Reference:
        .. [#] Wikipedia contributors. (2020, October 30). Arcsine distribution. https://en.wikipedia.org/w/index.php?title=Arcsine_distribution&oldid=986131091
    """

    def pdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]:
        """
        Args:
            x (Union[List[float], numpy.ndarray, float]): random variables

        Raises:
            ValueError: when there exist a value less than 0 or greater than 1
            TypeError: when parameter is not of type float | List[float] | numpy.ndarray

        Returns:
            Union[float, numpy.ndarray]: evaluation of pdf at x
        """
        if isinstance(x, (np.ndarray, List)):
            if not type(x) is np.ndarray:
                x = np.array(x)
            if np.any((x <= 0) | (x >= 1)):
                raise ValueError(
                    f'random variable should have values between [0,1].')
            return 1/(m.pi * np.sqrt(x*(1-x)))

        if type(x) is float:
            if x < 0 or x > 1:
                raise ValueError(
                    f'random variable should have values between [0,1].')
            return 1/m.pi*m.sqrt(x * (1-x))

        raise TypeError(
            'parameter x is expected to be of type float | List[float] | numpy.ndarray')

    def cdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]:
        """
        Args:
            x (Union[List[float], numpy.ndarray, float]): data point

        Raises:
            ValueError: when there exist a value less than 0 or greater than 1
            TypeError: when parameter is not of type float | List[float] | numpy.ndarray

        Returns:
            Union[float, numpy.ndarray]: evaluation of cdf at x
        """
        if isinstance(x, (np.ndarray, List)):
            if not type(x) is np.ndarray:
                x = np.array(x)
            if np.any((x <= 0) | (x >= 1)):
                raise ValueError(
                    f'values can only be evaluated in the domain [0,1]')
            return 1/(m.pi)*np.arcsin(np.sqrt(x))

        if type(x) is float:
            if x <= 0 or x >= 1:
                raise ValueError(
                    f'values can only be evaluated in the domain [0,1]')
            return 1/m.pi * m.asin(m.sqrt(x))

        raise TypeError(
            'parameter x is expected to be of type float | List[float] | numpy.ndarray')

    def mean(self) -> float:
        """
        Returns:
            mean of Arcsine distribution.
        """
        return 0.5

    def median(self) -> float:
        """
        Returns:
             median of Arcsine distribution
        """
        return 0.5

    def mode(self) -> Tuple[float, float]:
        """
        Returns:
            mode of Arcsine distribution
        """
        return (0, 1)

    def var(self) -> float:
        """
        Returns:
            variance of Arcsine distribution
        """
        return 0.125

    def std(self) -> float:
        """
        Returns:
            standard deviation of Arcsine distribution
        """
        return m.sqrt(0.125)

    def skewness(self) -> float:
        """
        Returns:
            skewness of Arcsine distribution
        """
        return 0.0

    def kurtosis(self) -> float:
        """
        Returns:
            kurtosis of Arcsine distribution
        """
        return 1.5

    def entropy(self) -> float:
        """
        Returns:
            entropy of Arcsine distribution
        """
        return m.log(m.pi/4)

    def summary(self) -> Dict[str, Union[float, Tuple[float, float]]]:
        """
        Returns:
            Dictionary of Arcsine distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }


class Beta(BoundedInterval):
    """
    This class contains methods concerning Beta Distirbution [#]_.

    .. math::
        \\text{Beta}(x; \\alpha, \\beta) = \\frac{x^{\\alpha-1}(1-x)^{\\beta-1}}{\\text{B}(\\alpha, \\beta)}

    Args:

        alpha(float): shape parameter where alpha > 0
        beta(float): shape parameter where beta > 0
        x(float): random variable where x is between 0 and 1

    Reference:
        .. [#] Wikipedia contributors. (2021, January 8). Beta distribution. https://en.wikipedia.org/w/index.php?title=Beta_distribution&oldid=999043368
    """

    def __init__(self, alpha: float, beta: float):
        if alpha < 0:
            raise ValueError(
                f'alpha parameter(shape) should be a positive number. Entered value:{alpha}')
        if beta < 0:
            raise ValueError(
                f'beta parameter(shape) should be a positive number. Entered value:{beta}')

        self.alpha = alpha
        self.beta = beta

    def pdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]:
        """
        Args:
            x (Union[List[float], numpy.ndarray, float]): random variable(s)

        Raises:
            ValueError: when there exist a value x <= 0 or x <= 1

        Returns:
            Union[float, numpy.ndarray]: evaluation of pdf at x
        """
        a = self.alpha
        b = self.beta

        if isinstance(x, (np.ndarray, List)):
            if not type(x) is np.ndarray:
                x = np.array(x)
            if np.any((x <= 0) | (x >= 1)):
                raise ValueError(
                    'random variables should only be between 0 and 1')
            return (np.power(x, a-1)*np.power(1-x, b-1))/ss.beta(a, b)

        if x <= 0 or x >= 1:
            raise ValueError('random variables should only be between 0 and 1')
        return (pow(x, a-1)*pow(1-x, b-1))/ss.beta(a, b)

    def cdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]:
        """
        Args:
            x (Union[List[float], numpy.ndarray]): data point(s) of interest

        Returns:
            Union[float, numpy.ndarray]: evaluation of cdf at x
        """
        a = self.alpha
        b = self.beta

        if isinstance(x, (np.ndarray, List)):
            if not type(x) is np.ndarray:
                x = np.array(x)
            return ss.betainc(a, b, x)

        return ss.betainc(a, b, x)

    def mean(self) -> str:
        """
        Returns: Mean of the Beta distribution.
        """
        return "currently unsupported."

    def median(self) -> float:
        """
        Returns: Median of the Beta distribution.
        """
        # warning: not yet validated.
        return ss.betainc(self.alpha, self.beta, 0.5)

    def mode(self) -> str:
        """
        Returns: Mode of the Beta distribution.
        """
        return "currently unsupported"

    def var(self) -> str:
        """
        Returns: Variance of the Beta distribution.
        """
        return "currently unsupported"

    def std(self) -> str:
        """
        Returns: Variance of the Beta distribution.
        """
        return "currently unsupported"

    def skewness(self) -> float:
        """
        Returns: Skewness of the Beta distribution.
        """
        alpha = self.alpha
        beta = self.beta
        return (2*(beta-alpha)*m.sqrt(alpha+beta+1))/((alpha+beta+2)*m.sqrt(alpha*beta))

    def kurtosis(self) -> float:
        """
        Returns: Kurtosis of the Beta distribution.
        """
        alpha = self.alpha
        beta = self.beta
        temp_up = 6*((alpha-beta)**2*(alpha+beta+1)-alpha*beta*(alpha+beta+2))
        return temp_up/(alpha*beta*(alpha+beta+2)*(alpha+beta+3))

    def entropy(self) -> float:
        """
        Returns: differential entropy of the Beta distribution.

        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier.
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        alpha = self.alpha
        beta = self.beta
        return m.log(ss.beta(alpha, beta))-(alpha-1)*(ss.digamma(alpha)-ss.digamma(alpha+beta))-(beta-1)*(ss.digamma(beta)-ss.digamma(alpha+beta))

    def summary(self) -> Dict[str, Union[float, str]]:
        """
        Returns:
            Dictionary of Beta distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }


class BetaRectangular(BoundedInterval):
    """
    This class contains methods concerning Beta-rectangular Distirbution.
    Thus it is a bounded distribution that allows for outliers to have a greater chance of occurring than does the beta distribution.

    .. math::
        \\text{BetaRectangulat}(x,\\alpha ,\\beta ,\\theta )={\\begin{cases}{\\frac{\\theta \\Gamma (\\alpha +\\beta )}{\\Gamma (\\alpha )\\Gamma (\\beta )}}{\\frac{(x-a)^{{\\alpha -1}}(b-x)^{{\\beta -1}}}{(b-a)^{{\\alpha +\\beta +1}}}}+{\\frac{1-\\theta }{b-a}}&{\mathrm{for}}\ a\leq x\leq b,\\\[8pt]0&{\\mathrm{for}}\ x<a\{\\mathrm{or}}\ x>b\\end{cases}}
    Args:

        alpha(float): shape parameter
        beta (float): shape parameter
        theta(float): mixture parameter where 0 < theta < 1
        min(float): lower bound
        max(float): upper bound
        x(float): random variable where alpha <= x<= beta

    Reference:
        .. [#] Wikipedia contributors. (2020, December 7). Beta rectangular distribution. https://en.wikipedia.org/w/index.php?title=Beta_rectangular_distribution&oldid=992814814
    """

    def __init__(self, alpha: float, beta: float, theta: float, min: float, max: float, randvar: float):
        if alpha < 0 or beta < 0:
            raise ValueError(
                'alpha and beta parameter should not be less that 0. Entered values: alpha: {alpha}, beta: {beta}}')
        if theta < 0 or theta > 1:
            raise ValueError(
                'random variable should only be in between 0 and 1. Entered value: {theta}')
        if randvar < min and randvar > max:
            raise ValueError(
                f'random variable should be between alpha and beta shape parameters. Entered value:{randvar}')

        self.alpha = alpha
        self.beta = beta
        self.theta = theta
        self.min = min
        self.max = max
        self.randvar = randvar

    def pdf(self, x: [List[float], np.ndarray, float]) -> Union[float,
                                                                 np.ndarray]: ...

    def cdf(self): ...

    def mean(self) -> float:
        """
        Returns: Mean of the Beta-rectangular distribution.
        """
        alpha = self.alpha
        beta = self.beta
        theta = self.theta
        a = self.min
        b = self.max
        return a+(b-a)*((theta*alpha)/(alpha+beta)+(1-theta)/2)

    def var(self) -> float:
        """
        Returns: Variance of the Beta-rectangular distribution.
        """
        alpha = self.alpha
        beta = self.beta
        theta = self.theta
        a = self.min
        b = self.max
        k = alpha+beta
        return (b-a)**2*((theta*alpha*(alpha+1))/(k*(k+1))+(1-theta)/3-(k+theta*(alpha-beta))**2/(4*k**2))

    def std(self) -> float:
        """
        Returns: Standard deviation of the Beta-rectangular distribution.
        """
        return m.sqrt(self.var())

    def summary(self) -> Dict[str, Union[float, Tuple[float]]]:
        """
        Summary statistic regarding the Beta-rectangular distribution which contains the following parts of the distribution:
        (mean, median, mode, var, std, skewness, kurtosis).
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }

class Bernoulli(BoundedInterval):
    """
    This class contains methods concerning Continuous Bernoulli Distirbution.
    The continuous Bernoulli distribution arises in deep learning and computer vision,
    specifically in the context of variational autoencoders, for modeling the
    pixel intensities of natural images [#]_ [#]_ [#]_ [#]_.

    .. math:: C(\\lambda)\\lambda^{x}(1-\\lambda)^{1-x}

    where 

    .. math:: C(\\lambda)= \\begin{cases}2&{\\text{if }\\lambda =\\frac {1}{2}} \\ \\frac{2\\tanh^{-1}(1-2\\lambda )}{1-2\\lambda }&{\\text{ otherwise}}\\end{cases}

    Args:

        shape(float): parameter
        x(float): random variable where x is between 0 and 1

    Reference:
        .. [#] Wikipedia contributors. (2020, November 2). Continuous Bernoulli distribution. https://en.wikipedia.org/w/index.php?title=Continuous_Bernoulli_distribution&oldid=986761458
        .. [#] Kingma, D. P., & Welling, M. (2013). Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114.
        .. [#] Kingma, D. P., & Welling, M. (2014, April). Stochastic gradient VB and the variational auto-encoder.In Second International Conference on Learning Representations, ICLR (Vol. 19).
        .. [#] Ganem, G & Cunningham, J.P. (2019). The continouous Bernoulli: fixing a pervasive error in variational autoencoders. https://arxiv.org/pdf/1907.06845.pdf
    """

    def __init__(self, shape: float):
        if shape < 0 or shape > 1:
            raise ValueError(
                'shape parameter a should only be in between 0 and 1.')

        self.shape = shape

    def pdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]:
        """
        Args:
            x (Union[List[float], numpy.ndarray, float]): random variable(s)

        Raises:
            ValueError: when there exist a value less than 0 or greater than 1

        Returns:
            Union[float, numpy.ndarray]: evaluation of cdf at x
        """

        shape = self.shape

        def __C(shape: float) -> float:
            return (2*m.atanh(1-2*shape)) / (1-2*shape) if shape != 0.5 else 2.0

        if isinstance(x, (np.ndarray, List)):
            if not type(x) is np.ndarray:
                x = np.array(x)
            if np.any((x <= 0)|(x >= 1)):
                raise ValueError('random variable must be between 0 and 1')
            return __C(self.shape) * np.power(shape, x)*np.power(1-shape, 1-x)

        if x <= 0 or x >= 1:
            raise ValueError('random variable must be between 0 and 1')
        return __C(self.shape)*pow(shape, x)*pow(1-shape, 1 - x)

    def cdf(self, x: Union[List[float], np.ndarray]) -> Union[float, np.ndarray]:
        """
        Args:
            x (Union[List[float], numpy.ndarray]): data point(s) of interest

        Raises:
            ValueError: when there exist a value <= 0 or >= 1

        Returns:
            Union[float, numpy.ndarray]: evaluation of cdf at x
        """
        shape = self.shape

        if isinstance(x, (np.ndarray, List)):
            if not type(x) is np.ndarray:
                x = np.array(x)
            if np.any((x <= 0)|(x >= 1)):
                raise ValueError('values must be between 0 and 1')
            return (np.power(shape, x)*np.power(1-shape, 1-x) + shape - 1)/(1-2*shape) if shape != 0.5 else x

        return (shape**x*pow(1-shape, 1-x)+shape-1)/(2*shape-1) if shape != 0.5 else x

    def mean(self) -> float:
        """
        Returns: Mean of the Continuous Bernoulli distribution.
        """
        shape = self.shape
        if shape == 0.5:
            return 0.5
        return shape/(2*shape-1)+(1/(2*np.arctanh(1-2*shape)))

    def var(self) -> float:
        """
        Returns: Variance of the Continuous Bernoulli distribution.
        """
        shape = self.shape
        if shape == 0.5:
            return 0.08333333333333333
        return shape/((2*shape-1)**2)+1/(2*np.arctanh(1-2*shape))**2

    def std(self) -> float:
        """
        Returns: Standard deviation of the Continuous Bernoulli distribution
        """
        return m.sqrt(self.var())

    def summary(self) -> Dict[str, Union[float, Tuple[float]]]:
        """
        Returns:
            Dictionary of Continuous Bernoulli distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }

class Bates(BoundedInterval):
    """
    This class contains methods concerning Bates Distirbution. Also referred to as the regular mean distribution.

    Note that the Bates distribution is a probability distribution of the mean of a number of statistically indipendent uniformly
    distirbuted random variables on the unit interval. This is often confused with the Irwin-Hall distribution which is
    the distribution of the sum (not the mean) of n independent random variables. The two distributions are simply versions of
    each other as they only differ in scale [#]_.

    
    Args:

        a(float): lower bound parameter 
        b(float): upper bound parameter where b > a
        n(int): where n >= 1 
        randvar(float): random variable where a <= x <= b

    Reference:
        .. [#] Wikipedia contributors. (2021, January 8). Bates distribution. https://en.wikipedia.org/w/index.php?title=Bates_distribution&oldid=999042206
    """

    def __init__(self, a: float, b: float, n: int, randvar: float):
        if randvar < 0 or randvar > 1:
            raise ValueError(
                f'random variable should only be in between 0 and 1. Entered value: {randvar}')
        if a > b:
            raise ValueError(
                'lower bound (a) should not be greater than upper bound (b).')
        if type(n) is not int:
            raise TypeError('parameter n should be an integer type.')

        self.a = a
        self.b = b
        self.n = n
        self.randvar = randvar

    def cdf(self, x: Union[List[float], np.ndarray] = None) -> Union[float, np.ndarray]:
        """
        Args:

            x (List[float], numpy.ndarray): random variable or list of random variables

        Returns:
            either cumulative distribution evaluation for some point or plot of Bates distribution.
        """
        return "currently unsupported"

    def mean(self) -> float:
        """
        Returns: Mean of the Bates distribution.
        """
        return 0.5*(self.a+self.b)

    def var(self) -> float:
        """
        Returns: Variance of the Bates distribution.
        """
        return 1/(12*self.n)*pow(self.b-self.a, 2)

    def std(self) -> float:
        """
        Returns: Standard devtiation of the Bates distribution
        """
        return m.sqrt(1/(12*self.n)*pow(self.b-self.a, 2))

    def skewness(self) -> float:
        """
        Returns: Skewness of the Bates distribution.
        """
        return -6/(5*self.n)

    def kurtosis(self) -> float:
        """
        Returns: Kurtosis of the Bates distribution.
        """
        return 0.0

    def summary(self) -> Dict[str, Union[float, Tuple[float]]]:
        """
        Summary statistic regarding the Bates distribution which contains the following parts of the distribution:
        (mean, median, mode, var, std, skewness, kurtosis).

        Returns:
            Dict[str, Union[float, Tuple[float]]]: [description]
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }


class Triangular(BoundedInterval):
    """
    This class contains methods concerning Triangular Distirbution [#]_.

    Args:

        a(float): lower limit parameter
        b(float): upper limit parameter where a < b
        c(float): mode parameter where a <= c <= b
        randvar(float): random variable where a <= x <= b

    Reference:
        .. [#] Wikipedia contributors. (2020, December 19). Triangular distribution. https://en.wikipedia.org/w/index.php?title=Triangular_distribution&oldid=995101682
    """

    def __init__(self, a: float, b: float, c: float):
        if a > b:
            raise ValueError(
                'lower limit(a) should be less than upper limit(b).')
        if a > c and c > b:
            raise ValueError(
                'lower limit(a) should be less than or equal to mode(c) where c is less than or equal to upper limit(b).')
        self.a = a
        self.b = b
        self.c = c

    def pdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]:
        """
        Args:
            x (Union[List[float], numpy.ndarray, float]): random variable(s)

        Raises:
            ValueError: when there exist a value of a > x or x > b 

        Returns:
            Union[float, numpy.ndarray]: evaluation of pdf at x
        """
        a, b, c = self.a, self.b, self.c

        def __generator(a: float, b: float, c: float, x: float) -> float:
            if x < a:
                return 0.0
            if a <= x and x < c:
                return (2*(x-a))/((b-a)*(c-a))
            if x == c:
                return 2/(b-a)
            if c < x and x <= b:
                return 2*(b-x)/((b-a)*((b-c)))
            if b < x:
                return 0.0

        if isinstance(x, (np.ndarray, List)):
            if not type(x) is np.ndarray:
                x = np.array(x)
            if np.any((a > x) | (x > b)):
                raise ValueError(
                    'all random variables are expected to be between a and b parameters')
            return np.vectorize(__generator)(a, b, c, x)

        if a > x or x > b:
            raise ValueError(
                'all random variables are expected to be between a and b parameters')

        return __generator(a, b, c, x)

    def cdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]:
        """
        Args:
            x (Union[List[float], numpy.ndarray, float]): data point(s) of interest

        Returns:
            Union[float, numpy.ndarray]: evaluation fo cdf at x
        """
        a, b, c = self.a, self.b, self.c

        def __generator(a: float, b: float, c: float, x: float) -> float:
            if x <= a:
                return 0.0
            if a < x and x <= c:
                return pow(x-a, 2)/((b-a)*(c-a))
            if c < x and x < b:
                return 1 - pow(b-x, 2)/((b-c)*(b-c))
            if b <= x:
                return 1.0

        if isinstance(x, (np.ndarray, List)):
            if not type(x) is np.ndarray:
                x = np.array(x)
            return np.vectorize(__generator)(a, b, c, x)

        return __generator(a, b, c, x)

    def mean(self) -> float:
        """
        Returns: Mean of the Triangular distribution.
        """
        return (self.a+self.b+self.c)/3

    def median(self) -> float:
        """
        Returns: Median of the Triangular distribution.
        """
        a = self.a
        b = self.b
        c = self.c
        if c >= (a+b)/2:
            return a + m.sqrt(((b-a)*(c-a))/2)
        if c <= (a+b)/2:
            return b + m.sqrt((b-a)*(b-c)/2)

    def mode(self) -> float:
        """
        Returns: Mode of the Triangular distribution.
        """
        return self.c

    def var(self) -> float:
        """
        Returns: Variance of the Triangular distribution.
        """
        a = self.a
        b = self.b
        c = self.c
        return (1/18)*(pow(a, 2)+pow(b, 2)+pow(c, 2)-a*b-a*c-b*c)

    def std(self) -> float:
        """
        Returns: Standard deviation of the Triangular distribution.
        """
        return m.sqrt(self.var())

    def skewness(self) -> float:
        """
        Returns: Skewness of the Triangular distribution.
        """
        a = self.a
        b = self.b
        c = self.c
        return m.sqrt(2)*(a+b-2*c) * ((2*a-b-c)*(a-2*b+c)) / \
            (5*pow(a**2+b**2+c**2-a*b-a*c-b*c, 3/2))

    def kurtosis(self) -> float:
        """
        Returns: Kurtosis of the Triangular distribution.
        """
        return -3/5

    def entropy(self) -> float:
        """
        Returns: differential entropy of the Triangular distribution.

        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier.
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        return 0.5 + m.log((self.b-self.a)*0.5)

    def summary(self) -> Dict[str, float]:
        """
        Returns:
            Dictionary of Triangular distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }


class LogitNormal(BoundedInterval):
    """
    This class contains methods concerning Logit Normal Distirbution [#]_.

    .. math::
        \\text{LogitNormal}(x;\\mu,\\sigma) = \\frac{1}{\\sigma \\sqrt(2\\pi) \\cdot x(1-x)} \\exp{\\Big(-\\frac{(logit(x)-\\mu)^2}{2\\sigma^2} \\Big)}

    Args:

        sq_scale (float): squared scale parameter
        location(float): location parameter
        x(float): random variable where x is between 0 and 1

    Reference:
        .. [#] Wikipedia contributors. (2020, December 9). Logit-normal distribution. https://en.wikipedia.org/w/index.php?title=Logit-normal_distribution&oldid=993237113
    """

    def __init__(self, sq_scale: float, location: float):
        self.sq_scale = sq_scale
        self.location = location

    def pdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]:
        """
        Args:
            x (Union[List[float], numpy.ndarray, float]): random variable(s)

        Raises:
            ValueError: when there exist a value below 0 and greater than 1

        Returns:
            Union[float, numpy.ndarray]: evaluation of pdf at x
        """
        mu = self.location
        sig = self.sq_scale

        if isinstance(x, (np.ndarray, List)):
            if not type(x) is np.ndarray:
                x = np.array(x)
            if np.any((x < 0) | (x > 1)):
                raise ValueError(
                    'random variable should only be in between 0 and 1')
            return (1/(sig*m.sqrt(2*m.pi))) * np.exp(-(np.power(ss.logit(x)-mu, 2)/(2*pow(sig, 2)))) * 1/(x*(1-x))

        if x < 0 or x > 1:
            raise ValueError(
                'random variable should only be in between 0 and 1')
        return (1/(sig*m.sqrt(2*m.pi))) * m.exp(-pow(ss.logit(x)-mu, 2)/(2*pow(sig, 2))) * 1/(x*(1-x))

    def cdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]:
        """
        Args:
            x (Union[List[float], numpy.ndarray, float]): data point(s) of interest

        Returns:
            Union[float, numpy.ndarray]: evaluation of cdf at x
        """
        mu = self.location
        sig = self.sq_scale

        def __generator(mu: float, sig: float, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
            return 0.5 * (1+ss.erf((ss.logit(x)-mu)/m.sqrt(2*pow(sig, 2))))

        if isinstance(x, (np.ndarray, List)):
            if not type(x) is np.ndarray:
                x = np.array(x)
            return __generator(mu, sig, x)

        return __generator(mu, sig, x)

    def mean(self) -> str:
        """
        Returns: Mean of the Logit Normal distribution.
        """
        return "no analytical solution"

    def mode(self) -> str:
        """
        Returns: Mode of the Logit Normal distribution.
        """
        return "no analytical solution"

    def var(self) -> str:
        """
        Returns: Variance of the Logit Normal distribution.
        """
        return "no analytical solution"

    def std(self) -> str:
        """
        Returns: Standard deviation of the Logit Normal distribution.
        """
        return "no analytical solution"

    def entropy(self) -> str:
        """
        Returns: differential entropy of Logit Normal distribution.

        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier.
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        return "unsupported"

    def summary(self) -> Dict[str, str]:
        """
        Returns:
            Dictionary of Logit Normal distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }


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

    def pdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]:
        """
        Args:
            x (Union[List[float], numpy.ndarray, float]): random variable(s)

        Returns:
            Union[float, numpy.ndarray]: evauation of pdf at x
        """
        a = self.a
        b = self.b

        if isinstance(x, (np.ndarray, List)):
            x0 = 1/(b-a)
            if not type(x) is np.ndarray:
                x = np.array(x)
            return np.piecewise(x, [(a <= x) & (x <= b), (a > x) | (x > b)], [x0, 0.0])

        return 1 / (b - a) if a <= x and x <= b else 0.0

    def cdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]:
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

        if isinstance(x, (np.ndarray, List)):
            if not type(x) is np.ndarray:
                x = np.array(x)
            # performance could be improved with np.piecewise
            return np.vectorize(__generator)(a, b, x)

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
        return m.sqrt(1 / 12 * pow(self.b - self.a, 2))

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
        return m.log(self.b-self.a)

    def summary(self) -> Dict[str, Union[float, Tuple[int, int]]]:
        """
        Returns:
            Dictionary of Uniform distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }


class Trapezoidal(BoundedInterval):
    """
    This class contains methods concerning Trapezoidal Distirbution [#]_.

    Args:

        a(float): lower bound parameter where a < d
        b(float): level start parameter where a <= b < c
        c(float): level end parameter where b < c <= d
        d(float): upper bound parameter where c <= d
        randvar(float): random variable where a <= x <= d 

    Reference:
        .. [#] Wikipedia contributors. (2020, April 11). Trapezoidal distribution. https://en.wikipedia.org/w/index.php?title=Trapezoidal_distribution&oldid=950241388
    """

    def __init__(self, a: float, b: float, c: float, d: float):
        if a > d:
            raise ValueError(
                'lower bound(a) should be less than upper bound(d).')
        if a > b or b >= c:
            raise ValueError(
                'lower bound(a) should be less then or equal to level start (b) where (b) is less than level end(c).')
        if b >= c or c > d:
            raise ValueError(
                'level start(b) should be less then level end(c) where (c) is less then or equal to upper bound (d).')
        if c > d:
            raise ValueError(
                'level end(c) should be less than or equal to upper bound(d)')

        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def pdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]:
        """
        Args:
            x (Union[List[float], numpy.ndarray, float]): random variable(s)

        Returns:
            Union[float, numpy.ndarray]: evaluation of pdf at x
        """
        a, b, c, d = self.a, self.b, self.c, self.d

        def __generator(a: float, b: float, c: float, d: float, x: float) -> float:
            if a <= x and x < b:
                return 2/(d+c-a-b) * (x-a)/(b-a)
            if b <= x and x < c:
                return 2/(d+c-a-b)
            if c <= x and x <= d:
                return (2/(d+c-a-b))*(d-x)/(d-c)

        if isinstance(x, (np.ndarray, List)):
            if not type(x) is np.ndarray:
                x = np.array(x)
            return np.vectorize(__generator)(a, b, c, d, x)

        return __generator(a, b, c, d, x)

    def cdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]:
        """

        Args:
            x (Union[List[float], numpy.ndarray, float]): data point(s) of interest

        Returns:
            Union[float, numpy.ndarray]: evaluation of cdf at x
        """
        a, b, c, d = self.a, self.b, self.c, self.d

        def __generator(a: float, b: float, c: float, d: float, x: float) -> float:
            if a <= x and x < b:
                return (x-a)**2/((b-a)*(d+c-a-b))
            if b <= x and x < c:
                return (2*x-a-b)/(d+c-a-b)
            if c <= x and x <= d:
                return 1 - (d-x)**2/((d+c-a-b)*(d-c))

        if isinstance(x, (np.ndarray, List)):
            if not type(x) is np.ndarray:
                x = np.array(x)
            return np.vectorize(__generator)(a, b, c, d, x)

        return __generator(a, b, c, d, x)

    def mean(self) -> float:
        """
        Returns: Mean of the Trapezoidal distribution.
        """
        a = self.a
        b = self.b
        c = self.c
        d = self.d

        return 1/(3*(d+c-b-a)) * ((d**3 - c**3)/(d-c) - (b**3 - a**3)/(b-a))

    def var(self) -> float:
        """
        Returns: Variance of the Trapezoidal distribution. Currently Unsupported.
        """
        a = self.a
        b = self.b
        c = self.c
        d = self.d

        mean = 1/(3*(d+c-b-a)) * ((d**3 - c**3)/(d-c) - (b**3 - a**3)/(b-a))
        return 1/(6*(d+c-b-a)) * ((d**4 - c**4)/(d-c) - (b**4 - a**4)/(b-a)) - pow(mean, 2)

    def summary(self) -> Dict[str, Union[float, Tuple[float]]]:
        """
        Returns:
            Dictionary of Trapezoidal distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }


class WignerSemiCircle(BoundedInterval): 

    def __init__(self, radius:float):
        if radius <= 0:
            raise ValueError('')

        self.radius = radius

    def pdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]: 
        
        rad = self.radius
        x0 = 2/(m.pi*rad**2)
        if isinstance(x, (List, np.ndarray)):
            if not type(x) is np.ndarray:
                x = np.array(x)

            # checks x <= R; x >= -R
            if np.any((x < -rad) | (x > rad)):
                raise ValueError(f'random variable is expected to be defined within [-{rad},{rad}]')
            return x0 * np.sqrt(rad**2 - np.power(x,2))
        
        if x < -rad or x > rad:
            raise ValueError(f'random variable is expected to be defined within [-{rad},{rad}]')
        return x0*m.sqrt(rad**2 - x**2)

    def cdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]: 

        rad = self.radius
        if isinstance(x, (List, np.ndarray)):
            if not type(x) is np.ndarray:
                x = np.array(x)

            # checks x <= R; x >= -R
            if np.any((x < -rad) | (x > rad)):
                raise ValueError(
                    f'data points are expected to be defined within [-{rad},{rad}]')
            return 0.5 + (x*np.sqrt(rad**2 - x**2))/(m.pi*rad**2) + np.arcsin(1/rad)/m.pi

        if x < -rad or x > rad:
            raise ValueError(
                f'data points are expected to be defined within [-{rad},{rad}]')
        return 0.5 + (x*m.sqrt(rad**2 - x**2))/(m.pi*rad**2) + m.asin(1/rad)/m.pi

    def mean(self)->float:
        return 0.0
    
    def median(self) -> float:
        return 0.0

    def mode(self) -> float: 
        return 0.0
    
    def var(self) -> float:
        return pow(self.radius, 4)

    def std(self)-> float:
        return pow(self.radius, 2)

    def skewness(self)-> float:
        return 0.0

    def kurtorsis(self) -> float:
        return -1.0
    
    def entropy(self) -> float:
        return m.log(m.pi* self.radius) * -0.5

    def summary(self) -> Dict[str, float]:
        """
        Returns:
            Dictionary of WignerSemiCircle distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }

# class IrwinHall(BoundedInterval): ...

class Kumaraswamy(BoundedInterval): 
    
    def __init__(self, a:float, b:float):
        if a <= 0 or b <= 0:
            raise ValueError('parameters are expected to have positive values')

        self.a, self.b = a,b
    
    def pdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]: 
        
        a,b = self.a, self.b

        if isinstance(x, (List, np.ndarray)):
            if not type(x) is np.ndarray:
                x = np.array(x)
            if np.any((x<=0) | (x >= 1)):
                raise ValueError('random variables are expected to be within (0,1)')
        else:
            if x <= 0 or x >= 1:
                raise ValueError('random variables are expected to be within (0,1)')
        
        return a*b*x**(a-1)*(1-x**a)**(b-1)

    def cdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]: 
        a,b = self.a, self.b

        if isinstance(x, (List, np.ndarray)):
            if not type(x) is np.ndarray:
                x = np.array(x)
            if np.any((x<=0) | (x >= 1)):
                raise ValueError('data points are expected to be within (0,1)')
        else:
            if x <= 0 or x >= 1:
                raise ValueError('data points are expected to be within (0,1)')
        
        return 1 - (1-x**a)**b
        

    def mean(self) -> float: 
        a,b = self.a, self.b
        return b*ss.gamma(1+1/a)*ss.gamma(b)/ss.gamma(1+1/a+b)

    def median(self) -> float: 
        return pow(1-pow(2,-1/self.b), 1/self.a)

    def mode(self) -> Union[float,str]: 
        a, b = self.a, self.b
        if a >= 1 and b >= 1:
            return pow((a-1)/(a*b-1), 1/a)
        return 'Undefined'

    def var(self) -> float: ...
    def std(self) -> float: ...
    def skewness(self) -> float: ...
    def kurtosis(self) -> float: ...

    def entropy(self) -> float: ...
    
    def summary(self) -> Dict[str, Union[float, str]]:
        """
        Returns:
            Dictionary of Kumaraswamy distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }


class Reciprocal(BoundedInterval): 
    
    def __init__(self, a:float, b:float):
        if a < 0 or b < 0:
            raise ValueError('parameters are expected to be greater than 0')
        if a >= b:
            raise ValueError('parameter a is expected to be less than b')
        
        self.a, self.b = a,b
    
    def pdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]: 
        a,b = self.a, self.b

        if isinstance(x, (List, np.ndarray)):
            if not type(x) is np.ndarray:
                x = np.array(x)

        return 1/(x*m.log(b/a))

    def cdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]: 
        a,b = self.a, self.b
        x0 = (m.log(b/a))
        if isinstance(x, (List, np.ndarray)):
            if not type(x) is np.ndarray:
                x = np.array(x)
            return np.log(x/a)/x0
        return m.log(x/a)/x0

    def mean(self) -> float: 
        a,b = self.a, self.b
        return (b-a)/m.log(b/a)

    def variance(self) -> float:
        a,b = self.a, self.b
        x0 = (b**2 - a**2)/(2*m.log(b/a))
        x1 = pow((b-a)/m.log(b/a),2)
        return x0 - x1

    def std(self) -> float: 
        return m.sqrt(self.var())

    def summary(self) -> Dict[str, float]:
        """
        Returns:
            Dictionary of Reciprocal distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }


class RaisedCosine(BoundedInterval): 
    def __init__(self, mu:float, s:float): 
        if s < 0:
            raise ValueError('parameter s is expected to be s <= 0')
        self.mu, self.s = mu, s

    def pdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]: 
        mu, s = self.mu, self.s
        l_bound, u_bound = mu-s, mu+s

        if isinstance(x, (List, np.ndarray)):
            if not type(x) is np.ndarray:
                x = np.array(x)            
            if np.any((x< l_bound) | (x>u_bound)):
                raise ValueError(f'random variables are expected to be in [{l_bound},{u_bound}]')
        else:
            if x < l_bound or x > u_bound:
                raise ValueError(f'random variables are expected to be in [{l_bound},{u_bound}]')
        return (1/2*s)*(1 + np.cos(m.pi*(x-mu)/s))

    def cdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]: 
        mu, s = self.mu, self.s

        if isinstance(x, (List, np.ndarray)):
            if not type(x) is np.ndarray:
                x = np.array(x)            
        return 0.5*(1 + (x-mu)/s + 1/m.pi*np.sin(m.pi*(x-mu)/s))

    def mean(self) -> float: 
        return self.mu

    def median(self) -> float: 
        return self.mu

    def mode(self) -> float: 
        return self.mu

    def var(self) -> float: 
        return self.s**2*(1/3 + 2/m.pi)

    def std(self) -> float: 
        return m.sqrt(self.s**2*(1/3 + 2/m.pi))

    def skewness(self) -> float: 
        return 0.0

    def kurtosis(self) -> float: 
        # 6*(90-m.pi**4)/(5*(m.pi**2 - 6)**2)
        return -0.5937628755982794

    def summary(self) -> Dict[str, float]:
        """
        Returns:
            Dictionary of Raised Cosine distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }


class UQuadratic(BoundedInterval): 
    
    def __init__(self, a:float, b:float): 
        if a >= b:
            raise ValueError('parameter a is expected to be less than b')
        
        self.a, self.b = a,b 
    
    def pdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]: 
        a,b = self.a, self.b
        midpoint = (a+b)/2

        if isinstance(x, (List, np.ndarray)):
            if not type(x) is np.ndarray:
                x = np.array(x)
            if np.any((x<a) | (x>b)):
                raise ValueError()
        else:
            if x < a or x  >b:
                raise ValueError()
        
        return a*(x - midpoint)**2 

    def cdf(self, x: Union[List[float], np.ndarray, float]) -> Union[float, np.ndarray]: 
        a,b = self.a, self.b
        midpoint = (a+b)/2

        if isinstance(x, (List, np.ndarray)):
            if not type(x) is np.ndarray:
                x = np.array(x)
            if np.any((x<a) | (x>b)):
                raise ValueError()
        else:
            if x < a or x  >b:
                raise ValueError()
        
        return a/3*((x-midpoint)**3 - (midpoint-a)**3)

    def mean(self) -> float: 
        return (self.a + self.b) / 2

    def median(self) -> float: 
        return (self.a + self.b) / 2

    def mode(self) -> Tuple[float, float]:
        return (self.a, self.b)

    def var(self) -> float: 
        return 3/20*pow(self.b - self.a, 2)

    def std(self) -> float: 
        return m.sqrt(3/20*pow(self.b - self.a, 2))

    def skewness(self) -> float: 
        return 0.0

    def kurtosis(self) -> float: 
        return 3/112*pow(self.b - self.a, 4)


# class PERT(BoundedInterval): ...


# class BaldingNichols(BoundedInterval): ...
