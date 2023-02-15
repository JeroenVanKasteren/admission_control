"""
Description of script...
"""

import numpy as np
# from numpy import array, arange, zeros, round
from numpy.random import default_rng as rng
from timeit import default_timer
# from numba import njit


class Env:
    """Class docstrings go here."""

    CONSTANTS = 1

    def __init__(self, **kwargs):  # **kwargs: Keyword arguments
        """Create all variables describing the environment."""
        self.alpha: float = kwargs.get('alpha')
        self.beta: float = kwargs.get('beta')
        self.B: int = kwargs.get('B')
        self.c_r: float = kwargs.get('c_r')  # rejection cost
        self.c_h: float = kwargs.get('c_h')  # holding cost
        self.gamma: float = kwargs.get('gamma')  # discount factor < 1

        self.lab: float = kwargs.get('lab', self.generate_lambda())
        self.mu: float = kwargs.get('mu', self.generate_lambda())

    def generate_lambda(self):
        """Generate lambda ~ Gamma(shape: k=alpha, scale: theta=1/beta) """
        return rng().gamma(self.alpha, 1/self.beta)

    def cost(self, x, event, tau, a):
        """Sample arrival (1) or departure (0)."""
        return (1/self.gamma * (1 - np.exp(-self.gamma*tau)) * x * self.c_h
                + np.exp(-self.gamma*tau) * event * (1 - a) * self.c_r)

    def transition(self, x, pi):
        """Sample arrival (1) or departure (0)."""
        event = self.event_sim(1)
        tau = self.time_sim(1)
        if isinstance(pi, int):  # Threshold value
            a = (x < self.B) and (x < pi)  # admit if True
        else:  # policy vector
            a = (x < self.B) and (pi[x] == 1)  # admit if True
        cost = self.cost(x, event, tau, a)
        x = max(x + event * (1 - a) - (1 - event), 0)
        return cost, self.x

    def event_sim(self, n):
        return rng().binomial(n=1, p=self.lab / (self.lab + self.mu), size=n)

    def time_sim(self, n):
        return rng().exponential(self.lab + self.mu, size=n)

    def lomax(s, x):
        """Docstring"""
        return s.alpha * s.beta ** s.aplha / (s.beta + x) ** (s.alpha + 1)

    def timer(self, name: str = "tmp", start: float = 0, trace: bool = True):
        """Only if trace=TRUE, start timer if start=true, else print time."""
        if not trace:
            pass
        elif start == 0:
            print('Starting ' + name + '.')
            return default_timer()
        else:
            time = default_timer() - start
            print("Time: ", int(time / 60), ":",
                  int(time - 60 * int(time / 60)))
            print('Finished ' + name + '.')
            return time
