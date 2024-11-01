import numpy as np
from numpy.random import default_rng as rng
import matplotlib.pyplot as plt
from scipy.special import gamma as gamma_fun
import mpmath as mp

N = 10000


def gamma_dist(a, b):
    print(f'Gamma distribution with mean={a/b:.3} and variance={a/(b*b):.3}')
    gamma_draws = rng().gamma(a, 1/b, N)
    _, x, _ = plt.hist(gamma_draws, 50, density=True)
    y = b**a * x**(a-1) * (np.exp(-b*x) / gamma_fun(a))
    plt.plot(x, y, linewidth=2, color='r')
    plt.show()


a, b = 5, 6
gamma_dist(a, b)

mp.gammainc(-5, a=1, b=mp.mpf("inf"))
