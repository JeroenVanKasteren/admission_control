"""
Description of script...
"""

import numpy as np
from utils.env import Env

class PolicyIteration:
    """Policy Iteration"""

    def __init__(self, env: Env):
        """Docstring goes here."""
        self.name = 'Policy Iteration'
        self.v = None  # V_{t-1}
        self.pi = None
        self.n_iter = 0
        self.converged = False

    @staticmethod
    def init_pi(env: Env, threshold=False):
        """Initialize policy.
        Forces pi[0] = 1 (admit) and pi[B+1] = 0 (reject)"""
        individual = max(1, min(env.B - 1, env.lab * env.c_r / env.c_h))
        if threshold:
            pi = np.arange(env.B + 1) < individual
        else:
            pi = np.ceil(individual)
        return pi

    @staticmethod
    def get_v(env: Env, v, pi=None):
        """Calculate V_{t+1}.
        Value Iteration: forces admit when empty and reject when full.
        Policy evaluation: assumes admit when empty and reject when full."""
        y_dep = [0] + list(range(env.B))
        v_t = env.c_h * np.arange(env.B + 1) + env.mu * v[y_dep]

        y_arr = list(range(1, env.B))
        if pi is None:  # Value iteration
            v_t[0] += env.lab * v[1]
            v_t[1:-1] += env.lab * np.minimum(v[y_arr], env.c_r + v[:-1])
            v_t[-1] += env.lab * (env.c_r + v[-1])
        else:  # Policy Evaluation
            y_arr = y_arr + [env.B]  # add last state
            v_t += env.lab * (pi * v[y_arr]
                              + (1 - pi) * (env.c_r + v))
        return v_t / (env.gamma + env.lab + env.mu)

    # @staticmethod
    # def get_v_threshold(env: Env, v, pi):
    #     """Calculate V_{t+1}.
    #     assumes admit when empty and reject when full."""
    #     v_t = np.zeros(env.B+1)
    #     y_dep = np.maximum(np.arange(-1, pi+1), 0)
    #     v_t[:pi+1] = (env.c_h * np.arange(pi+1)
    #                   + env.lab * v[1:pi+1]
    #                   + env.mu * v[y_dep])
    #     y_arr = np.minimum(np.arange(pi, env.B + 1) + 1, env.B)
    #     v_t[pi+1:] = (env.c_h * np.arange(pi, env.B + 1)
    #                   + env.lab * (env.c_r + v[y_arr])
    #                   + env.mu * v[pi:env.B])
    #     return v_t / (env.gamma + env.lab + env.mu)

    @staticmethod
    def convergence(env: Env, v_t, v, i, name, j=-1):
        """Convergence check of valid states only."""
        diff = v_t - v
        delta_max = max(diff)
        delta_min = min(diff)
        converged = delta_max - delta_min < env.eps
        max_iter = (i > env.max_iter) | (j > env.max_iter)
        if ((converged & env.trace) |
                (env.trace & (i % env.print_modulo == 0 |
                               j % env.print_modulo == 0))):
            print("iter: ", i,
                  "inner_iter: ", j,
                  ", delta: ", round(delta_max - delta_min, 2),
                  ', d_min', round(delta_min, 2),
                  ', d_max', round(delta_max, 2))
        elif converged:
            if j == -1:
                print(name, 'converged in', i, 'iterations.')
            else:
                print(name, 'converged in', j, 'iterations.')
        elif max_iter:
            print(name, 'iter:', i, 'reached max_iter =', max_iter)
        return converged, max_iter

    @staticmethod
    def policy_improvement(env: Env, v, pi):
        """Determine best action/policy per state by one-step lookahead.
        assumes and keeps pi[0] = 1 (admit) and pi[B+1] = 0 (reject)
        """
        pi_t = pi.copy()
        pi_t[1:env.B] = np.argmin(v[2:env.B + 1], env.c_r + v[1:env.B])
        return pi_t, (pi_t == pi).all()

    @staticmethod
    def policy_improvement_threshold(env: Env, v, pi):
        """Determine best action/policy per state by one-step lookahead.
        Forces pi[0] = 1 (admit) and pi[B+1] = 0 (reject)
        """
        y_arr = list(range(1, env.B + 1)) + [env.B]
        pi_t =  max(1, min(env.B - 1, np.argmax(env.c_r + v < v[y_arr])))
        return pi_t, pi_t == pi

    @staticmethod
    def policy_evaluation(env: Env, v,  pi, name, n_iter=0):
        """Policy Evaluation."""
        assert pi is not None, 'Policy not initialized.'
        inner_iter = 0
        stopped = False
        converged = False
        while not (stopped | converged):
            v_t = PolicyIteration.get_v(env, v, pi)
            if inner_iter % env.convergence_check == 0:
                converged, stopped = PolicyIteration.convergence(
                    env, v_t, v, n_iter, name, j=inner_iter)
            v = v_t - v_t[0]  # Rescale and Save v_t
            inner_iter += 1
        return v, converged, inner_iter

    def policy_iteration(s, env: Env, **kwargs):
        """Docstring."""
        s.v = kwargs.get('v', np.zeros(env.B + 1, dtype=np.float64))  # V_{t-1}
        s.pi = kwargs.get('pi', s.init_pi(env))
        max_pi_iter = kwargs.get('max_pi_iter', env.max_iter)

        while not s.converged:
            s.v, _, _ = s.policy_evaluation(env, s.v, s.pi,
                                            'Policy Evaluation of PI',
                                            s.n_iter)
            s.pi, s.converged = s.policy_improvement(env, s.v, s.pi)

            if s.n_iter > max_pi_iter:
                print(f'Policy Iteration reached max_iter ({max_pi_iter}).')
                break
            s.n_iter += 1
