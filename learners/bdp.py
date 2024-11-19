"""
Description of script...
"""

import numpy as np
from utils.env import Env
from learners import PolicyIteration
import mpmath as mp

class BDP:
    """Bayesian Dynamic Programming."""

    def __init__(self, env: Env, **kwargs):
        """Docstring goes here."""
        self.name = 'Bayesian Dynamic Programming'
        self.d = env.alpha/env.beta  # Init on average rate
        self.delta = kwargs.get('delta', 20)
        self.D = kwargs.get('D', self.delta * 5)
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
        """Discount factor for rate costs: alpha=k+1, beta=kd/delta."""
        return (-(beta * (env.mu + env.gamma)) ^ alpha
                   * (env.alpha * mp.gammainc(-alpha,
                                              a=beta * (env.mu + env.gamma),
                                              b=mp.mpf("inf"))
                      + env.mu / (env.mu + env.gamma)
                      * mp.gammainc(1 - alpha,
                                    a=beta * (env.mu + env.gamma),
                                    b=mp.mpf("inf"))))

    def gamma_lump(self, env: Env, k, d, d_n):
        """Discount factor for rate costs."""
        return np.exp(-env.gamma * ((k + 1) * d_n - k * d) / self.delta)

    def get_v(self, env, x, k, d, v):
        """Calculate V_{t+1}."""
        gamma_r = self.gamma_rate(env, k + 1, k * d / self.delta)
        v_state = gamma_r * x * env.c_h
        v_a = np.zeros(2)
        for d_n in range(1, self.D + 1):
            for a in range(env.A):
                v_a[a] = (self.gamma_lump(env, k, d, d_n)
                          * self.p_trans(env, k, d, d_n)
                          * ((1 - a) * env.c_r
                             + v[min(x + a, env.B),
                                min(k + 1, self.K),
                                d_n]))
        v_state += min(v_a)
        return v_state, np.argmin(v_a)

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
                        v_state, _ = self.get_v(x, k, d)
                        v_t[x, k, d] += v_state
            if n_iter % env.convergence_check == 0:
                converged, stopped = PolicyIteration.convergence(
                    env, v_t, self.v, n_iter, self.name)
            v = v_t - v_t[0, 0, 1]  # Rescale v_t
            n_iter += 1

    def learn(self, env: Env):
        """Bayesian Dynamic Programming learns everything in advance."""
        pass

    def choose(self, env: Env):
        """Choose an action."""
        # Calculate discretized d based on k/t
        d = round(env.k / env.t / self.delta)
        _, a = self.get_v(env.x[env.t], env.k, d)
        return a
