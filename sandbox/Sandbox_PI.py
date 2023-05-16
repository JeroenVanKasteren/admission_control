"""
Sandbox Policy Iteration
"""

import numpy as np
from utils.Env import Env
from sandbox.Sandbox_VI import convergence, policy_improvement
from types import SimpleNamespace
np.set_printoptions(precision=4, linewidth=150, suppress=True)

env = Env(alpha=5, beta=6, mu=5, B=100,
          c_r=-10, c_h=-1, gamma=-np.log(0.99),
          print_modulo=100, seed=42)

self = {'name': 'Policy Iteration',
        'v': np.zeros(env.B + 1, dtype=float),  # v_{t-1}
        'pi': np.array([1]*env.B + [0], dtype=np.int32),
        'threshold': env.B - 1,
        'stable': False,
        'count': 0}

self = SimpleNamespace(**self)


# staticmethod
def policy_evaluation(s, env):
    """Policy Evaluation."""
    inner_count = 0
    stopped = False
    converged = False
    x = np.arange(env.B + 1)
    y_dep = [0] + list(range(env.B))
    while not (stopped | converged):
        v_t = (env.c_h * x
               + env.lab * ((1 - s.pi) * env.c_r + s.v[x + s.pi])
               + env.mu * s.v[y_dep]) / (env.gamma + env.lab + env.mu)
        if inner_count % env.convergence_check == 0:
            converged, stopped = convergence(env, v_t, s.v, s.count,
                                               s.name, j=inner_count)
        s.v = v_t - v_t[0]  # Rescale v_t
        if inner_count > env.max_iter:
            return s, converged
        inner_count += 1
    return s, converged


# staticmethod
def policy_iteration(s, env):
    """Policy iteration."""
    s.stable = False
    stopped = False
    while not (stopped | s.stable):
        s, stopped = policy_evaluation(s, env)
        s, s.stable = policy_improvement(s, env)
        print(f'Threshold: {s.threshold}')
        if s.count > env.max_iter:
            break
        s.count += 1
    return s


if __name__ == "__main__":
    self = policy_iteration(self, env)
    print(self.threshold, self.pi)
