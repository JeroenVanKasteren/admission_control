"""
Description of script...
"""

import numpy as np
from learners import PolicyIteration
from utils.env import Env

class ValueIteration:
    """Value Iteration."""

    def __init__(self, env,
                 method='certainty_equivalent',
                 max_iter=np.inf,
                 eps=1e-3,
                 print_modulo=np.inf,  # 1 for always
                 convergence_check=1e0):
        """Docstring goes here."""
        self.name = 'Value Iteration'
        self.method = method
        self.lab = env.lab if method == 'full_info' else env.alpha/env.beta
        self.pi_learner = PolicyIteration(name=self.name,
                                          max_iter=max_iter,
                                          eps=eps,
                                          print_modulo=print_modulo,
                                          convergence_check=convergence_check)

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
            v_t = self.pi_learner.get_v(env, v)
            if n_iter % self.pi_learner.convergence_check == 0:
                converged, stopped = self.pi_learner.convergence(v_t, v, n_iter)
            v = v_t - v_t[0]  # Rescale v_t
            n_iter += 1
        return v

    def learn(self, env: Env):
        """Update V-value using value iteration."""
        if self.method == 'full_info':
            return
        lab_t = env.t / env.k
        update = np.abs(self.lab - lab_t)/self.lab > self.pi_learner.eps
        if update:
            self.lab = lab_t
            self.value_iteration(env)

    def choose(self, env: Env):
        """Choose an action."""
        x = env.x[env.t]
        return env.c_r + self.v[x] < self.v[min(x + 1, env.B)]
