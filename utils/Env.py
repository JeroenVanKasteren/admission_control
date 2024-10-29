"""
Description of script...
"""

import numpy as np
from timeit import default_timer
from time import perf_counter as clock, strptime


class Env:
    """Class docstrings go here."""

    CONSTANTS = 1

    def __init__(self, **kwargs):  # **kwargs: Keyword arguments
        """Create all variables describing the environment."""
        self.rng = np.random.default_rng(kwargs.get('seed', 42))
        self.alpha: float = kwargs.get('alpha')
        self.beta: float = kwargs.get('beta')
        self.B: int = kwargs.get('B')  # Buffer size
        self.c_r: float = kwargs.get('c_r')  # Rejection cost
        self.c_h: float = kwargs.get('c_h')  # Holding cost
        self.gamma: float = kwargs.get('gamma')  # Discount factor < 1
        self.eps: float = kwargs.get('eps', 1e-4)

        self.mu: float = kwargs.get('mu')
        self.lab: float = kwargs.get('lab', self.generate_lambda())

        self.max_iter = kwargs.get('max_iter', np.Inf)
        self.start_time = clock()
        if 'max_time' in kwargs:  # max time in seconds, 60 seconds slack
            x = strptime(kwargs.get('max_time'), '%H:%M:%S')
            self.max_time = x.tm_hour * 60 * 60 + x.tm_min * 60 + x.tm_sec - 60
        else:
            self.max_time = np.Inf
        self.print_modulo = kwargs.get('print_modulo', np.inf)  # 1 for always
        self.convergence_check = kwargs.get('convergence_check', 1)
        print(f'alpha = {self.alpha} beta = {self.beta}, B = {self.B}, \n'
              f'gamma = {self.gamma}, c_r = {self.c_r}, c_h = {self.c_h}, \n'
              f'mu = {self.mu}, lab = {self.lab}, \n')

    def generate_lambda(self):
        """Generate lambda ~ Gamma(shape: k=alpha, scale: theta=1/beta) """
        return self.rng.gamma(self.alpha, 1/self.beta)

    def cost(self, x, event, tau, a):
        """Given event (arrival/departure) and time tau, output the cost."""
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

    def time_print(self, time):
        """Convert seconds to readable format."""
        print(f'Time: {time/60:.0f}:{time - 60 * int(time / 60):.0f} min.\n')
