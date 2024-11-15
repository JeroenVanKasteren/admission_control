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
        self.A: int = 2  # Action size
        self.c_r: float = kwargs.get('c_r')  # Rejection cost
        self.c_h: float = kwargs.get('c_h')  # Holding cost
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

        # initialize state
        self.x, self.a, self.r, self.t = self.reset()

    @staticmethod
    def reset():
        """Reset the environment to the initial state."""
        return [0], [], [], 0  # x, a, r, t

    def generate_lambda(self):
        """Generate lambda ~ Gamma(shape: k=alpha, scale: theta=1/beta) """
        return self.rng.gamma(self.alpha, 1/self.beta)

    def event_sim(self, n):
        return self.rng.binomial(n=1, p=self.lab / (self.lab + self.mu), size=n)

    def time_sim(self, n):
        return self.rng.exponential(self.lab + self.mu, size=n)

    def get_return(self, x, event, tau, a):
        """Given event (arrival/departure) and time tau, output the cost."""
        return ((1 - np.exp(-self.gamma*tau)) * x * self.c_h / self.gamma
                + np.exp(-self.gamma*tau) * event * (1 - a) * self.c_r)

    def n_step_return(self, gamma, n, t):
        """Compute the expected return, G_{t:t+n}."""
        return sum([gamma ** (k - 1) * self.r[t+k] for k in range(1, n)])

    def transition(self, pi):
        """Sample arrival (1) or departure (0)."""
        x = self.x[self.t]  # current state
        event = self.event_sim(1)
        tau = self.time_sim(1)
        if isinstance(pi, int):  # Threshold value
            a = (x < self.B) and (x < pi)  # admit if True
        else:  # policy vector
            a = (x < self.B) and (pi[x] == 1)  # admit if True
        self.r.append(self.get_return(x, event, tau, a))
        self.x.append(max(x + event * (1 - a) - (1 - event), 0))
        return event

    def lomax(s, x):
        """Docstring"""
        return s.alpha * s.beta ** s.aplha / (s.beta + x) ** (s.alpha + 1)
