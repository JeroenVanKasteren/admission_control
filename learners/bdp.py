"""
Description of script...
"""

import numpy as np
from utils.env import Env
import mpmath as mp

class BDP:
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
            l = ((d_n - 1 / 2) * (k + 1) - k * d) / self.delta
            u = ((d_n + 1 / 2) * (k + 1) - k * d) / self.delta
        else:
            l = k * (d_n - 1 / 2 - d) / self.delta
            u = k * (d_n + 1 / 2 - d) / self.delta
        l_tilde = env.mu * env.beta * (l / env.beta + 1)
        u_tilde = env.mu * env.beta * (u / env.beta + 1)
        term1 = (env.mu * env.beta) ^ env.alpha * np.exp(env.mu * env.beta)
        term_l = (1 + l / env.beta) ^ (-env.alpha) * np.exp(-env.mu * l)
        term_u = (1 + u / env.beta) ^ (-env.alpha) * np.exp(-env.mu * u)
        if arr:
            if d_n < self.D:  # gammainc(z, a=L, b=U)
                return (-term1 * mp.gammainc(1 - env.alpha,
                                             a=l_tilde,
                                             b=u_tilde)
                        + term_l - term_u)
            else:  # d_n = D
                return (-term1 * mp.gammainc(1 - env.alpha,
                                             a=l_tilde,
                                             b=mp.mpf("inf"))
                        + term_l)
        else:  # dep
            if d_n < self.D:
                return (-env.alpha * term1 * mp.gammainc(-env.alpha,
                                                         a=l_tilde,
                                                         b=u_tilde)
                        + term_l - term_u)
            else:  # d_n = D
                return (-env.alpha * term1 * mp.gammainc(1 - env.alpha,
                                                         a=l_tilde,
                                                         b=mp.mpf("inf"))
                        + term_l)

    def gamma_discount(self, env: Env, d_n):
        k, d = self.state['k'], self.state['d']
        gamma = np.exp(-env.gamma * ((k + 1) * d_n - k * d) / self.delta)
        gamma_h = (-(env.beta * (env.mu + env.gamma)) ^ env.alpha
                   * (env.alpha * mp.gammainc(-env.alpha,
                                              a=env.beta * (env.mu + env.gamma),
                                              b=mp.mpf("inf"))
                      + env.mu / (env.mu + env.gamma)
                      * mp.gammainc(1 - env.alpha,
                                    a=env.beta * (env.mu + env.gamma),
                                    b=mp.mpf("inf"))))
        return gamma, gamma_h

