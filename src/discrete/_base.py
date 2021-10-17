class Base:  
    def __init__(self):
        if type(self) is Base:
            raise TypeError('Discrete Univariate Base class cannot be instantiated.')

    def pvalue(self) -> NotImplemented:
        """
        Default implementation of p-value.
        Returns NotImplemented.
        """
        return NotImplemented

    def confidence_interval(self) -> NotImplemented:
        """
        Default implementation of confidence interval.
        Returns NotImplemented.
        """
        return NotImplemented

    def rvs(self):  # (adaptive) rejection sampling implementation
        """
        returns random variate samples default NotImplemented
        """
        return "currently unsupported"

    def mean(self) -> NotImplemented:
        """
        Default implementation of the mean.
        Returns NotImplemented.
        """
        return NotImplemented

    def median(self) -> NotImplemented:
        """
        Default implementation of the median.
        Returns NotImplemented.
        """
        return NotImplemented

    def mode(self) -> NotImplemented:
        """
        Default implementation of the mode.
        Returns NotImplemented.
        """
        return NotImplemented

    def var(self) -> NotImplemented:
        """
        Default implementation of the variance.
        Returns NotImplemented.
        """
        return NotImplemented

    def std(self) -> NotImplemented:
        """
        Default implementation of the standard deviation.
        Returns NotImplemented.
        """
        return NotImplemented

    def skewness(self) -> NotImplemented:
        """
        Default implementation of skewness.
        Returns NotImplemented.
        """
        return NotImplemented

    def kurtosis(self) -> NotImplemented:
        """
        Default implementation of kurtosis.
        Returns NotImplemented.
        """
        return NotImplemented

    def entropy(self) -> NotImplemented:
        """
        Default implementation of entropy.
        Returns NotImplemented.
        """
        return NotImplemented

class Finite(Base):
    """
    Description:
        Base class for probability tags.
    """
    def __init__(self):
        if type(self) is Infinite:
            raise TypeError('base class cannot be instantiated.')

class Infinite(Base):
    """
    Description:
        Base class for probability tags.
    """
    def __init__(self):
        if type(self) is Infinite:
            raise TypeError('base class cannot be instantiated.')