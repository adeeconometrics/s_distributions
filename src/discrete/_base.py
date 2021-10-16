class Base:  
    def __init__(self):
        if type(self) is Base:
            raise TypeError('Discrete Univariate Base class cannot be instantiated.')

    def pvalue(self) -> NotImplemented:
        return NotImplemented

    def confidence_interval(self) -> NotImplemented:
        return NotImplemented

    def rvs(self):  # (adaptive) rejection sampling implementation
        """
        returns random variate samples default NotImplemented
        """
        return "currently unsupported"

    def mean(self) -> NotImplemented:
        """
        returns mean default NotImplemented
        """
        return NotImplemented

    def median(self) -> NotImplemented:
        """
        returns median default NotImplemented
        """
        return NotImplemented

    def mode(self) -> NotImplemented:
        """
        returns mode default NotImplemented
        """
        return NotImplemented

    def var(self) -> NotImplemented:
        """
        returns variance default NotImplemented
        """
        return NotImplemented

    def std(self) -> NotImplemented:
        """
        returns the std default (undefined)
        """
        return NotImplemented

    def skewness(self) -> NotImplemented:
        """
        returns skewness default NotImplemented
        """
        return NotImplemented

    def kurtosis(self) -> NotImplemented:
        """
        returns kurtosis default NotImplemented
        """
        return NotImplemented

    def entropy(self) -> NotImplemented:
        """
        returns entropy default NotImplemented
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