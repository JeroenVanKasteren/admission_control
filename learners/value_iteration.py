"""
Description of script...
"""

import numpy as np
from learners import PolicyIteration as pi_learner
from utils.env import Env

class ValueIteration:
    """Value Iteration."""

    def __init__(self, env, method='certainty_equivalent', eps=1e-2):
        """Docstring goes here."""
        self.name = 'Value Iteration'
        self.method = method
        self.lab = env.lab if method == 'full_info' else env.alpha/env.beta
        self.eps = eps

        self.v = self.value_iteration(env)

    # @staticmethod
    # def get_v_threshold(env: Env, v):
    #     """Calculate V_{t+1}."""
    #     y_arr = list(range(1, env.B + 1)) + [env.B]
    #     pi = np.argmax(env.c_r + v < v[y_arr])
    #     return pi_learner.get_v_threshold(env, v, pi)

    def value_iteration(self, env: Env):
        v = np.zeros([env.B + 1])  # V_{t-1}
        converged = False
        stopped = False
        n_iter = 0
        while not (stopped | converged):  # Update each state.
            v_t = pi_learner.get_v(env, self.v)
            if n_iter % env.convergence_check == 0:
                converged, stopped = pi_learner.convergence(
                    env, v_t, self.v, n_iter, self.name + ' ' + self.method)
            self.v = v_t - v_t[0]  # Rescale v_t
            n_iter += 1
        return v

    def learn(self, env: Env):
        """Update V-value using value iteration."""
        if self.method == 'full_info':
            return
        lab_t = env.t / env.k
        update = np.abs(self.lab - lab_t)/self.lab > self.eps
        if update:
            self.lab = lab_t
            self.value_iteration(env)

    def choose(self, env: Env, t):
        """Choose an action."""
        x = env.x[env.t]
        return env.c_r + self.v[x] < self.v[min(x + 1, env.B)]
