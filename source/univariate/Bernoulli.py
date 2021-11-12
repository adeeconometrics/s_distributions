try:
    import numpy as _np
    from typing import Union, Tuple, Dict, List
    from math import sqrt as _sqrt, atanh as _atanh
    from univariate._base import BoundedInterval
except Exception as e:
    print(f"some modules are missing {e}")


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
            raise ValueError('shape parameter a should only be in between 0 and 1.')

        self.shape = shape

    def pdf(self, x: Union[List[float], _np.ndarray, float]) -> Union[float, _np.ndarray]:
        """
        Args:
            x (Union[List[float], _np.ndarray, float]): random variable(s)

        Raises:
            ValueError: when there exist a value less than 0 or greater than 1
            TypeError: when parameter is not of type float | List[float] | numpy.ndarray

        Returns:
            Union[float, _np.ndarray]: evaluation of cdf at x
        """

        shape = self.shape
        
        def __C(shape: float)->float:
            return (2*_atanh(1-2*shape)) / (1-2*shape) if shape != 0.5 else 2.0

        
        if isinstance(x, (_np.ndarray, List)):
            x = _np.fromiter(x, dtype=float)
            if _np.any(_np.logical_or(x<=0, x>=1)):
                raise ValueError('random variable must be between 0 and 1')
            return __C(self.shape) * _np.power(shape, x)*_np.power(1-shape, 1-x)

        if type(x) is float:
            if x<=0 or x>=1:
                raise ValueError('random variable must be between 0 and 1')
            return __C(self.shape)*pow(shape, x)*pow(1-shape, 1 - x)

        raise TypeError(f'parameter x is expected to be of type float | List[float] | numpy.ndarray')

    def cdf(self, x: Union[List[float], _np.ndarray]) -> Union[float, _np.ndarray]:
        """
        Args:
            x (Union[List[float], _np.ndarray]): data points of interest

        Raises:
            ValueError: when there exist a value <= 0 or >= 1
            TypeError: when parameter is not of type float | List[float] | numpy.ndarray

        Returns:
            Union[float, _np.ndarray]: evaluation of cdf at x
        """
        shape = self.shape

        if isinstance(x, (_np.ndarray, List)):
            x = _np.fromiter(x, dtype=float)
            if _np.any(_np.logical_or(x<=0, x>=1)):
                raise ValueError('values must be between 0 and 1')
            return (_np.power(shape, x)*_np.power(1-shape, 1-x) + shape - 1)/(1-2*shape) if shape != 0.5 else x

        if type(x) is float:
            return (shape**x*pow(1-shape, 1-x)+shape-1)/(2*shape-1) if shape != 0.5 else x

        raise TypeError(f'parameter x only accepts List types or numpy.ndarray')

    def mean(self) -> float:
        """
        Returns: Mean of the Continuous Bernoulli distribution.
        """
        shape = self.shape
        if shape == 0.5:
            return 0.5
        return shape/(2*shape-1)+(1/(2*_np.arctanh(1-2*shape)))

    def var(self) -> float:
        """
        Returns: Variance of the Continuous Bernoulli distribution.
        """
        shape = self.shape
        if shape == 0.5:
            return 0.08333333333333333
        return shape/((2*shape-1)**2)+1/(2*_np.arctanh(1-2*shape))**2

    def std(self) -> float:
        """
        Returns: Standard deviation of the Continuous Bernoulli distribution
        """
        return _sqrt(self.var())

    def summary(self) -> Dict[str, Union[float, Tuple[float]]]:
        """
        Returns:
            Dictionary of Continuous Bernoulli distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }
