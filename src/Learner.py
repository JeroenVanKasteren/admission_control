"""
Description of script...
"""

import numpy as np
from numpy import array, arange, round
# from numba import njit
from src.Env import Env
from scipy.special import gamma as gamma_fun, gammaincc as reg_up_inc_gamma


class ValueIteration:
    """Name of Learner."""

    def __init__(self, env: Env):
        """Docstring goes here."""
        self.name = 'Value Iteration'
        self.v = np.zeros(env.B + 1, dtype=float)  # V_{t-1}
        self.pi = np.append(np.ones(env.B, dtype=float), 0)
        self.g = 0
        self.count = 0

    def function_name(self, param):
        """Docstring goes here."""

    @staticmethod
    def calculate_v(env, v):
        """Calculate V_{t+1}."""
        y_dep = [0] + list(range(env.B))
        y_arr = list(range(1, env.B+1)) + [env.B]
        c_r_v = array([0]*env.B + [env.c_r])
        v_t = (env.c_h * arange(env.B + 1)
               + env.lab * np.minimum(v[y_arr] + c_r_v, env.c_r + v)
               + env.mu * v[y_dep])
        return v_t / (env.gamma + env.lab + env.mu)

    def value_iteration(s, env):
        """Docstring."""
        start = env.timer(name=s.name)
        while True:  # Update each state.
            v_t = s.calculate_v(env, s.v)
            converged, s.g = s.convergence(env, v_t, s.v, s.count, s.name)
            if converged:
                break  # Stopping condition
            s.v = v_t - v_t[0]  # Rescale and Save v_t
            s.count += 1
        env.timer(name=s.name, start=start)

    @staticmethod
    def convergence(env, v_t, v, i, name, j=-1):
        """Convergence check of valid states only."""
        diff = v_t - v
        delta_max = max(diff)
        delta_min = min(diff)
        converged = delta_max - delta_min < env.eps
        max_iter = (i > env.max_iter) | (j > env.max_iter)
        g = (delta_max + delta_min) / 2 * (env.lab + env.mu)
        if ((converged & env.trace) |
                (env.trace & (i % env.print_modulo == 0 |
                               j % env.print_modulo == 0))):
            print("iter: ", i,
                  "inner_iter: ", j,
                  ", delta: ", round(delta_max - delta_min, 2),
                  ', D_min', round(delta_min, 2),
                  ', D_max', round(delta_max, 2),
                  ", g: ", round(g, 4))
        elif converged:
            if j == -1:
                print(name, 'converged in', i, 'iterations. g=', round(g, 4))
            else:
                print(name, 'converged in', j, 'iterations. g=', round(g, 4))
        elif max_iter:
            print(name, 'iter:', i, 'reached max_iter =', max_iter, ', g~',
                  round(g, 4))
        return converged | max_iter, g

    def policy(self, env, PI_learner):
        """Determine policy via Policy Improvement."""
        self.Pi, _ = PI_learner.policy_improvement(self.V, self.Pi, True,
                                                   env.J, env.S, env.D, env.gamma, env.t, env.c, env.r, env.P,
                                                   env.size, env.sizes, env.S_states, env.x_states, env.dim, env.P_xy,
                                                   PI_learner.NONE_WAITING, PI_learner.KEEP_IDLE,
                                                   PI_learner.SERVERS_FULL)


class PolicyIteration:
    """Policy Iteration."""

    def __init__(self, env):
        self.name = 'Policy Iteration'
        self.v = np.zeros(env.B + 1, dtype=float)  # V_{t-1}
        self.pi = np.append(np.ones(env.B, dtype=float), 0)
        self.g = 0
        self.count = 0
        self.stable = True

    @staticmethod
    def v_threshold(env, v, pi):
        """Calculate V_{t+1}."""
        v_t = np.zeros(env.B+1)
        y_dep = np.maximum(arange(-1, pi+1), 0)
        v_t[:pi+1] = (env.c_h * arange(pi+1)
                      + env.lab * v[1:pi+1]
                      + env.mu * v[y_dep])
        y_arr = np.minimum(arange(pi, env.B + 1) + 1, env.B)
        v_t[pi+1:] = (env.c_h * arange(pi, env.B + 1)
                      + env.lab * (env.c_r + v[y_arr])
                      + env.mu * v[pi:env.B])
        return v_t / (env.gamma + env.lab + env.mu)

    @staticmethod
    def policy_improvement_threshold(env, v, pi):
        """Determine best action/policy per state by one-step lookahead.

        assume pi[0] = 1 (admit) and pi[B+1] = 0 (reject)
        """
        pi_t = pi.copy()  # error?
        # choose maximum policy
        pi_t[1:env.B] = np.argmin(v[2:env.B+1], env.c_r + v[1:env.B])
        return pi_t, (pi_t == pi).all()

    @staticmethod
    def policy_improvement(env, v, pi):
        """Determine best action/policy per state by one-step lookahead.

        assume pi[0] = 1 (admit) and pi[B+1] = 0 (reject)
        """
        pi_t = pi.copy()
        pi_t[1:env.B] = np.argmin(v[2:env.B+1], env.c_r + v[1:env.B])
        return pi_t, (pi_t == pi).all()

    def policy_evaluation(self, env, V, W, Pi, name, count=0):
        """Policy Evaluation."""
        inner_count = 0
        converged = False
        while not converged:
            W = self.init_W(env, V, W)
            W = self.W_f(V, W, Pi, env.J, env.D, env.gamma, env.t, env.c, env.r,
                         env.size, env.size_i, env.sizes, env.sizes_i,
                         env.dim_i,env.s_states, env.x_states, env.P_xy)
            V_t = self.V_f(env, V, W)
            converged, g = self.convergence(env, V_t, V, count, name, j=inner_count)
            V = V_t - V_t[tuple([0] * (env.J * 2))]  # Rescale and Save V_t
            inner_count += 1
        return V, g

    def policy_iteration(self, env):
        """Docstring."""
        env.timer(True, self.name, env.trace)
        while self.stable:
            V = zeros(env.dim)  # V_{t-1}
            self.g, self.V, self.W = self.policy_evaluation(env, self.V, self.W, self.Pi, 'Policy Evaluation of PI',
                                                            self.count)
            self.Pi, self.unstable = self.policy_improvement(self.V, self.Pi, self.unstable,
                                                             env.J, env.S, env.D, env.gamma, env.t, env.c, env.r, env.P,
                                                             env.size, env.sizes, env.S_states, env.x_states, env.dim,
                                                             env.P_xy,
                                                             self.NONE_WAITING, self.KEEP_IDLE, self.SERVERS_FULL)
            if (self.count > env.max_iter) | (not self.unstable):
                print(self.name, 'converged in', self.count, 'iterations. g=', round(self.g, 4))
            if self.count > env.max_iter:
                break
            self.count += 1
        env.timer(False, self.name, env.trace)