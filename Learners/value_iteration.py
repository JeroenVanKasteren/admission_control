"""
Description of script...
"""

import numpy as np
from utils.Env import Env
from types import SimpleNamespace
from time import perf_counter as clock
import sys

class ValueIteration:
    """Name of Learner."""

    def __init__(self, env: Env):
        """Docstring goes here."""
        self.name = 'Value Iteration'
        self.v = np.zeros(env.B + 1, dtype=float)  # V_{t-1}
        self.pi = np.array([1]*env.B + [0], dtype=np.int32)
        self.count = 0
        self.theta = 0

    # @staticmethod
    # def convergence(env, v_t, v, i, name, j=-1):
    #     """Convergence check."""
    #     diff = v_t - v
    #     delta_max = max(diff)
    #     delta_min = min(diff)
    #     converged = delta_max - delta_min < env.eps
    #     max_iter = (i > env.max_iter) | (j > env.max_iter)
    #     time = (clock() - env.start_time)
    #     max_time = time > env.max_time
    #     if (converged | (((i % env.print_modulo == 0) & (j == -1))
    #                      | (j % env.print_modulo == 0))):
    #         print(f'iter: {i}, inner_iter: {j}, '
    #               f'delta: {delta_max - delta_min:.3f}, '
    #               f'd_min: {delta_min:.3f}, d_max: {delta_max:.3f}')
    #     if converged:
    #         iter = i if j == -1 else j
    #         print(f'{name} converged in {iter} iterations.\n')
    #     elif max_iter:
    #         print(f'{name} iter {i}, ({j}) reached max_iter ({max_iter})\n')
    #     elif max_time:
    #         print(f'{name} iter {i}, ({j}) reached max_time ({max_time})\n')
    #     return converged, max_iter | max_time
    #
    # def value_iteration(s, env):
    #     """Value iteration."""
    #     x = np.arange(env.B + 1)
    #     y_dep = [0] + list(range(env.B))
    #     stopped = False
    #     while not (stopped | s.converged):
    #         s.v_t[0] = env.lab * s.v[1]
    #         s.v_t[1:env.B] = env.lab * np.minimum(s.v[2:env.B + 1],
    #                                               env.c_r + s.v[1:env.B])
    #         s.v_t[env.B] = env.lab * (env.c_r + s.v[env.B])
    #         s.v_t += env.c_h * x + env.mu * s.v[y_dep]
    #         s.v_t = s.v_t / (env.gamma + env.lab + env.mu)
    #         if s.count % env.convergence_check == 0:
    #             s.converged, stopped = s.convergence(env, s.v_t, s.v, s.count,
    #                                                s.name)
    #         s.v = s.v_t - s.v_t[0]  # Rescale V_t
    #         s.count += 1
    #
    # def get_policy(self, env):
    #     """Get policy."""
    #     self.pi[1:env.B] = np.argmin([self.v[2:env.B + 1],
    #                                   env.c_r + self.v[1:env.B]], axis=0)
    #     self.theta = np.nonzero(self.pi == 0)[0][0]
    #     return self.theta
