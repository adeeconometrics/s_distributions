try:
    from scipy.special import logit as _logit, erf as _erf
    import numpy as _np
    from math import sqrt as _sqrt, pi as _pi, exp as _exp
    from typing import Union, Tuple, Dict, List
    from univariate._base import BoundedInterval
except Exception as e:
    print(f"some modules are missing {e}")


class LogitNormal(BoundedInterval):
    """
    This class contains methods concerning Logit Normal Distirbution.
    Args:

        sq_scale (float): squared scale parameter
        location(float): location parameter
        randvar(float | [0,1]): random variable

    Reference:
    - Wikipedia contributors. (2020, December 9). Logit-normal distribution. In Wikipedia, The Free Encyclopedia.
    Retrieved 07:44, December 30, 2020, from https://en.wikipedia.org/w/index.php?title=Logit-normal_distribution&oldid=993237113
    """

    def __init__(self, sq_scale: Union[float, int], location: Union[float, int], randvar: Union[float, int]):
        if randvar < 0 or randvar > 1:
            raise ValueError(
                f'random variable should only be in between (0,1). Entered value: randvar:{randvar}')
        self.sq_scale = sq_scale
        self.location = location
        self.randvar = randvar

    def pdf(self, x: Union[List[float], _np.ndarray] = None) -> Union[float, _np.ndarray]:
        """
        Args:

            x (List[float], numpy.ndarray): random variable or list of random variables

        Returns:
            either probability density evaluation for some point or plot of Logit Normal distribution.
        """
        mu = self.location
        sig = self.sq_scale 
        randvar = self.randvar

        if x is not None:
            if not isinstance(x, (_np.ndarray, List)):
                raise TypeError(f'parameter x only accepts List types or numpy.ndarray')
            else:
                x = _np.array(x)
                return (1/(sig*_sqrt(2*_pi)))* _np.exp(-(_np.power(_logit(x)-mu, 2)/(2*pow(sig, 2)))) * 1/(x*(1-x))
        
        return (1/(sig*_sqrt(2*_pi)))* _exp(-pow(_logit(x)-mu, 2)/(2*pow(sig, 2))) * 1/(x*(1-x))

    def cdf(self, x: Union[List[float], _np.ndarray] = None) -> Union[float, _np.ndarray]:
        """
        Args:

            x (List[float], numpy.ndarray): random variable or list of random variables

        Returns:
            either cumulative distribution evaluation for some point or plot of Logit Normal distribution.
        """
        mu = self.location
        sig = self.sq_scale 
        randvar = self.randvar

        def __generator(mu:float, sig:float, x:Union[float, _np.ndarray]) -> Union[float, _np.ndarray]:
            return 1/2 * (1+_erf((_logit(x)-mu)/_sqrt(2*pow(sig, 2))))

        if x is not None:
            if not isinstance(x, (_np.ndarray, List)):
                raise TypeError(f'parameter x only accepts List types or numpy.ndarray')
            else:
                x = _np.array(x)
                return __generator(mu, sig, x)

        return __generator(mu, sig, randvar)

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
        Summary statistic regarding the LogitNormal distribution which contains the following parts of the distribution:
        (mean, median, mode, var, std, skewness, kurtosis).

        Returns:
            Dict[str, str]
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }
