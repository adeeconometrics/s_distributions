from univariate._base import Base
import matplotlib.pyplot as plt
import numpy as np


def plot(Distribution:Base, x:np.ndarray)->None:
    # plot multiple versions of the distirbution in 
    # different parameterization. Annotate vaules on
    # legend and include annotations on the image.
    plt.plot(x,Distribution.pdf(x))
    plt.savefig('../docs/img/' + type(Distribution).__name__ + 'PDF.png')

# unpack arguments and set constructor values on tuple

if __name__ == "__main__":
    from univariate.Infinite import Gaussian
    N = Gaussian()
    plot(N, np.linspace(-1,1,1000))