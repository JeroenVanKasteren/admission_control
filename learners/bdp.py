"""
Description of script...
"""
from idlelib.sidebar import LineNumbers

import numpy as np
from utils.env import Env

class bdp:
    """Bayesian Dynamic Programming."""

    def __init__(self, env: Env, **kwargs):
        """Docstring goes here."""
        self.name = 'Bayesian Dynamic Programming'
        self.state = {'k': 0, 'd': env.alpha/env.beta}
        self.delta = kwargs.get('delta', 25)
        self.D = kwargs.get('D', self.delta*100)

    def p_trans(self, env: Env, d_n, arr=True):
        l = self.state['k'] * (d_n - 1/2 - self.state['d']) / self.delta
        u = self.state['k'] * (d_n + 1/2 - self.state['d']) / self.delta
        l_tilde = env.mu * env.beta * (l / env.beta + 1)
        u_tilde = env.mu * env.beta * (u / env.beta + 1)
        term1 = (env.mu * env.beta) ^ env.alpha * np.exp(env.mu * env.beta)
        term2 = (1 + l / env.beta) ^ (-env.alpha) * np.exp(-env.mu * l)
        term3 = (1 + u / env.beta) ^ (-env.alpha) * np.exp(-env.mu * u)
        if arr:
            if d_n < self.D:
                return (-term1 * (self.lower_gamma(1 - env.alpha, u_tilde)
                                 - self.lower_gamma(1 - env.alpha, l_tilde))
                        + term2 - term3)
            else:  # d_n = D
                return (-term1 * self.upper_gamma(1 - env.alpha, l_tilde)
                        - term2)
        else:  # dep
            if d_n < self.D:
                return (-env.alpha * term1
                        * (self.lower_gamma(-env.alpha, u_tilde)
                           - self.lower_gamma(-env.alpha, l_tilde))
                        + term2 - term3)
            else:  # d_n = D
                return (term1 * self.upper_gamma(1 - env.alpha, l_tilde)
                        - term2)