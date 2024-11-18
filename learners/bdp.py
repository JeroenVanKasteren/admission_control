"""
Description of script...
"""
from importlib.metadata import pass_none

import numpy as np
from utils.env import Env
import mpmath as mp

class BDP:
    """Bayesian Dynamic Programming."""

    def __init__(self, env: Env, **kwargs):
        """Docstring goes here."""
        self.name = 'Bayesian Dynamic Programming'
        self.state = {'d': env.alpha/env.beta}  # Initial state, get k from env
        self.delta = kwargs.get('delta', 25)
        self.D = kwargs.get('D', self.delta * 100)
        self.K = kwargs.get('K', 100)

        self.value_iteration(env)

    def p_trans(self, env: Env, k, d, d_n, arr=True):
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

    @staticmethod
    def gamma_rate(env: Env, alpha, beta):
        return (-(beta * (env.mu + env.gamma)) ^ alpha
                   * (env.alpha * mp.gammainc(-alpha,
                                              a=beta * (env.mu + env.gamma),
                                              b=mp.mpf("inf"))
                      + env.mu / (env.mu + env.gamma)
                      * mp.gammainc(1 - alpha,
                                    a=beta * (env.mu + env.gamma),
                                    b=mp.mpf("inf"))))

    def gamma_lump(self, env: Env, k, d, d_n):
        return np.exp(-env.gamma * ((k + 1) * d_n - k * d) / self.delta)

    def value_iteration(self, env: Env):
        v = np.zeros([env.B + 1, self.K + 1, self.D])  # V_{t-1}
        v_t = np.zeros([env.B + 1, self.K + 1, self.D])  # V_t
        converged = False
        stopped = False
        n_iter = 0
        while not (stopped | converged):  # Update each state.
            for x in range(env.B + 1):
                for k in range(self.K + 1):
                    for d in range(1, self.D + 1):
                        gamma_r = self.gamma_rate(env, k, d)
                        v_t[x, k, d] = gamma_r * x * env.c_h
                        for d_n in range(1, self.D + 1):
                            gamma_l = self.gamma_lump(env, k, d, d_n)
                            p_trans = self.p_trans(env, k, d, d_n)
                            # v_admit = gamma_l * p_trans * (
                            #     (1 - a) * env.c_r + v[min(x + a, env.B), k + 1, d_n]
                            # v_t[x, k, d] += min(v_reject, v_admit)
            if n_iter % env.convergence_check == 0:
                converged, stopped = pi_learner.convergence(
                    env, v_t, self.v, n_iter, self.name + ' ' + self.method)
            v = v_t - v_t[0, 0, 0]  # Rescale v_t
            n_iter += 1

    def learn(self, env: Env):
        """Update state using Bayesian Dynamic Programming."""
        pass

    def choose(self, env: Env):
        """Choose an action."""
        x = env.x[env.t]
        v_reject = 0
        v_admit = 0
        for d_n in range(1, self.D + 1):
            self.gamma_discount(env, d_n)
        return (env.c_r + self.v[x, env.k + 1, d_n]
                < self.v[min(x + 1, env.B), env.k + 1, d_n])