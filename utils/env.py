"""
Description of script...
"""

import numpy as np


class Env:
    """Class docstrings go here."""

    CONSTANTS = 1

    def __init__(self, **kwargs):  # **kwargs: Keyword arguments
        """Create all variables describing the environment."""
        # Fixed parameters
        self.alpha: float = kwargs.get('alpha')
        self.beta: float = kwargs.get('beta')
        self.B: int = int(kwargs.get('B'))  # Buffer size
        self.A: int = 2  # Action size
        self.c_r: float = kwargs.get('c_r')  # Rejection cost
        self.c_h: float = kwargs.get('c_h')  # Holding cost
        self.gamma: float = kwargs.get('gamma')  # discount factor
        self.eps: float = kwargs.get('eps', 1e-4)
        self.rho: float = kwargs.get('rho')

        self.steps = kwargs.get('steps', 1e5)
        self.x = self.a = self.r = self.k = self.t = self.time = None
        self.lab = self.mu = None
        self.events = self.times = None

        print(f'alpha = {self.alpha} beta = {self.beta}, B = {self.B}, \n'
              f'gamma = {self.gamma}, c_r = {self.c_r}, c_h = {self.c_h}, \n'
              f'rho = {self.rho}.\n')

    def reset(self, seed=42, **kwargs):
        """Reset the environment to the initial state."""
        self.lab: float = kwargs.get('lab', self.generate_lambda(seed))
        if 'mu' in kwargs:
            self.mu: float = kwargs.get('mu', self.mu)
        else:
            self.mu: float = self.lab / self.rho
        print(f'mu = {self.mu}, lab = {self.lab}, rho = {self.rho}.\n')

        surplus = 10 * self.mu / self.lab
        self.events = self.event_sim(int(surplus * self.steps), seed=seed)
        self.times = self.time_sim(int(surplus * self.steps), seed=seed)

        self.x = [0]  # [x_0]
        self.a = []  # a_t
        self.r = [0]  # [r_0]
        self.k = 0  # k
        self.t = 0  # t (steps)
        self.time = 0.  # time
    def generate_lambda(self, seed=42):

        """Generate lambda ~ Gamma(shape: k=alpha, scale: theta=1/beta) """
        rng = np.random.default_rng(int(seed))
        return rng.gamma(self.alpha, 1/self.beta)

    def event_sim(self, n, seed=42):
        rng = np.random.default_rng(int(seed + self.t))
        return rng.binomial(n=1, p=self.lab / (self.lab + self.mu), size=n)

    def time_sim(self, n, seed=42):
        rng = np.random.default_rng(int(seed + self.t))
        return rng.exponential(self.lab + self.mu, size=n)

    def get_return(self, x, event, tau, a):
        """Given event (arrival/departure) and time tau, output the cost."""
        return ((1 - np.exp(-self.gamma*tau)) * x * self.c_h / self.gamma
                + np.exp(-self.gamma*tau) * event * (1 - a) * self.c_r)

    def n_step_return(self, gamma, n, t):
        """Compute the expected return, G_{t:t+n}."""
        return sum([gamma ** (k - 1) * self.r[t+k] for k in range(1, n)])

    def step(self, a):
        """Sample arrival (1) or departure (0)."""
        while True:
            x = self.x[self.t]  # current state, x_t
            event = self.events[self.t]
            tau = self.times[self.t]
            self.a.append(a)  # a_t
            self.r.append(self.get_return(x, event, tau, a))  # r_{t+1}
            self.x.append(max(x + event * (1 - a) - (1 - event), 0))  # x_{t+1}
            self.k += event
            self.t += 1
            self.time += tau
            if event == 1:
                return

    def lomax(s, x):
        """Docstring"""
        return s.alpha * s.beta ** s.aplha / (s.beta + x) ** (s.alpha + 1)
