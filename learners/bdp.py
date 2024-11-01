"""
Description of script...
"""
from idlelib.sidebar import LineNumbers

import numpy as np
from utils.env import Env
import mpmath as mp

class bdp:
    """Bayesian Dynamic Programming."""

    def __init__(self, env: Env, **kwargs):
        """Docstring goes here."""
        self.name = 'Bayesian Dynamic Programming'
        self.state = {'k': 0, 'd': env.alpha/env.beta}
        self.delta = kwargs.get('delta', 25)
        self.D = kwargs.get('D', self.delta*100)

    def p_trans(self, env: Env, d_n, arr=True):
        k, d = self.state['k'], self.state['d']
        if arr:
            l = k * (d_n - 1/2 - d) / self.delta
            u = k * (d_n + 1/2 - d) / self.delta
        else:
            l = ((d_n - 1 / 2) * (k + 1) - k * d) / self.delta
            u = ((d_n + 1 / 2) * (k + 1) - k * d) / self.delta
        l_tilde = env.mu * env.beta * (l / env.beta + 1)
        u_tilde = env.mu * env.beta * (u / env.beta + 1)
        term1 = (env.mu * env.beta) ^ env.alpha * np.exp(env.mu * env.beta)
        term2 = (1 + l / env.beta) ^ (-env.alpha) * np.exp(-env.mu * l)
        term3 = (1 + u / env.beta) ^ (-env.alpha) * np.exp(-env.mu * u)
        if arr:
            if d_n < self.D:
                # mp.gammainc(-5, a=1, b=mp.mpf("inf")) TODO: implement
                return (-term1 * self.g(env.alpha, l_tilde, u_tilde)
                        + term2 - term3)
            else:  # d_n = D
                return (-term1 * self.g(env.alpha, l_tilde)
                        + term2)
        else:  # dep
            if d_n < self.D:
                return (-env.alpha * term1
                        * self.g(env.alpha + 1, l_tilde, u_tilde)
                        + term2 - term3)
            else:  # d_n = D
                return (-env.alpha * term1 * self.g(env.alpha + 1, l_tilde)
                        + term2)

