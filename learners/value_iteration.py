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
                 eps_update = 2e-2,  # percentage change in lab
                 print_modulo=np.inf,  # 1 for always
                 convergence_check=1e0):
        """Docstring goes here."""
        self.name = 'Value Iteration'
        self.method = method
        self.lab = env.lab if method == 'full_info' else env.alpha/env.beta
        self.eps_update = eps_update
        self.pi_learner = PolicyIteration(name=self.name,
                                          max_iter=max_iter,
                                          eps=eps,
                                          print_modulo=print_modulo,
                                          convergence_check=convergence_check)
        self.v = np.zeros([env.B + 1])
        self.value_iteration(env, trace=True)

    # @staticmethod
    # def get_v_threshold(env: Env, v):
    #     """Calculate V_{t+1}."""
    #     y_arr = list(range(1, env.B + 1)) + [env.B]
    #     pi = np.argmax(env.c_r + v < v[y_arr])
    #     return pi_learner.get_v_threshold(env, v, pi)

    def value_iteration(self, env: Env, trace=False):
        # Init v = 0, or use previous v if update
        converged = False
        stopped = False
        n_iter = 0
        while not (stopped | converged):  # Update each state.
            v_t = self.pi_learner.get_v(env, self.v)
            if n_iter % self.pi_learner.convergence_check == 0:
                converged, stopped = self.pi_learner.convergence(v_t,
                                                                 self.v,
                                                                 n_iter,
                                                                 trace=trace)
            self.v = v_t - v_t[0]  # Rescale v_t
            n_iter += 1

    def learn(self, env: Env):
        """Update V-value using value iteration."""
        if self.method == 'full_info':
            return
        lab_t = env.t / env.k
        update = np.abs(self.lab - lab_t)/self.lab > self.eps_update
        if update:
            self.lab = lab_t
            self.value_iteration(env)

    def choose(self, env: Env):
        """Choose an action."""
        x = env.x[env.t]
        return  self.v[min(x + 1, env.B)] < env.c_r + self.v[x]

    def print_pi(self, env: Env):
        """Get policy."""
        x = np.arange(env.B)
        pi_new = self.v[x + 1] < env.c_r + self.v[x]
        if not np.all(pi_new == self.pi_learner.pi):
            if np.any(pi_new == False):
                threshold = np.argmax(1 - pi_new)
            else:
                threshold = env.B
            if np.all(pi_new[threshold:] == False):
                print('time ', env.t, 'threshold:', threshold)
            else:
                print('time ', env.t, 'policy:', pi_new)
        self.pi_learner.pi = pi_new