"""
Description of script...
"""

import numpy as np
from numpy import array, arange, zeros, round
# from numba import njit
from src.Env import Env
from scipy.special import gamma as gamma_fun, gammaincc as reg_up_inc_gamma


class ValueIteration:
    """Name of Learner."""

    def __init__(self, env: Env):
        """Docstring goes here."""
        self.name = 'Value Iteration'
        self.V = zeros(env.B, dtype=float)  # V_{t-1}
        self.Pi = zeros(env.B, dtype=float)
        self.g = 0
        self.count = 0

    def function_name(self, param):
        """Docstring goes here."""


    @staticmethod
    @njit
    def W_f(V, W, J, S, D, gamma, t, c, r, P, size, sizes, S_states, x_states, dim, P_xy):
        """W."""
        V = V.reshape(size);
        W = W.reshape(size)
        for s in S_states:
            for x in x_states:
                state = np.sum(x * sizes[0:J] + s * sizes[J:J * 2])
                W[state] = V[state]
                if np.sum(s) < S:
                    W[state] = W[state] - P if np.any(x == D) else W[state]
                    for i in arange(J):
                        if (x[i] > 0):  # If someone of class i waiting
                            value = r[i] - c[i] if x[i] > gamma * t[i] else r[i]
                            for y in arange(x[i] + 1):
                                next_x = x.copy()
                                next_x[i] = y
                                next_s = s.copy()
                                next_s[i] += 1
                                next_state = np.sum(next_x * sizes[0:J] + \
                                                    next_s * sizes[J:J * 2])
                                value += P_xy[i, x[i], y] * V[next_state]
                            W[state] = array([value, W[state]]).max()
        return W.reshape(dim)

    def value_iteration(self, env):
        """Docstring."""
        env.timer(True, self.name, env.trace)
        while True:  # Update each state.
            V_t = PI_learner.V_f(env, self.V, self.W)
            converged, self.g = PI_learner.convergence(env, V_t, self.V, self.count, self.name)
            if (converged):
                break  # Stopping condition
            # Rescale and Save V_t
            self.V = V_t - V_t[tuple([0] * (env.J * 2))]
            self.count += 1
        env.timer(False, self.name, env.trace)

    def policy(self, env, PI_learner):
        """Determine policy via Policy Improvement."""
        self.Pi, _ = PI_learner.policy_improvement(self.V, self.Pi, True,
                                                   env.J, env.S, env.D, env.gamma, env.t, env.c, env.r, env.P,
                                                   env.size, env.sizes, env.S_states, env.x_states, env.dim, env.P_xy,
                                                   PI_learner.NONE_WAITING, PI_learner.KEEP_IDLE,
                                                   PI_learner.SERVERS_FULL)

