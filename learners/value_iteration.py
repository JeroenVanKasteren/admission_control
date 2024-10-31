"""
Description of script...
"""

import numpy as np
from learners import PolicyIteration as pi_learner
from utils.env import Env

class ValueIteration:
    """Value Iteration."""

    def __init__(self):
        """Docstring goes here."""
        self.name = 'Value Iteration'
        self.v = None
        self.pi = None
        self.n_iter = 0
        self.converged = False

    @staticmethod
    def get_v_threshold(env: Env, v):
        """Calculate V_{t+1}."""
        y_arr = list(range(1, env.B + 1)) + [env.B]
        pi = np.argmax(env.c_r + v < v[y_arr])
        return pi_learner.get_v_threshold(env, v, pi)

    def value_iteration(s, env: Env, **kwargs):
        s.v = kwargs.get('v', np.zeros(env.B + 1, dtype=np.float64))  # V_{t-1}
        stopped = False
        while not (stopped | s.converged):  # Update each state.
            v_t = pi_learner.get_v(env, s.v)
            if s.n_iter % env.convergence_check == 0:
                s.converged, stopped = pi_learner.convergence(
                    env, v_t, s.v, s.n_iter, s.name)
            s.v = v_t - v_t[0]  # Rescale V_t
            s.n_iter += 1
