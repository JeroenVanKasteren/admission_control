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
        self.v = zeros(env.B, dtype=float)  # V_{t-1}
        self.pi = zeros(env.B, dtype=float)
        self.g = 0
        self.count = 0

    def function_name(self, param):
        """Docstring goes here."""

    def calculate_v(env, v):
        """Calculate V_{t+1}."""
        y_dep = [0] + list(range(env.B))
        y_arr = list(range(1, env.B+1)) + [env.B]
        c_r_v = array([0]*env.B + [env.c_r])
        v_t = (env.c_h * arange(env.B + 1)
               + env.lab * np.minimum(v[y_arr] + c_r_v, env.c_r + v)
               + env.mu * v[y_dep])
        return v_t / (env.gamma + env.lab + env.mu)

    def value_iteration(self, env):
        """Docstring."""
        env.timer(True, name=self.name, trace=env.trace)
        while True:  # Update each state.
            V_t = self.calculate_v(env, self.V, self.W)
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

