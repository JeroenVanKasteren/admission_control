"""
Sandbox Value Iteration
"""

import numpy as np
from utils.Env import Env
from types import SimpleNamespace
from time import perf_counter as clock

np.set_printoptions(precision=4, linewidth=150, suppress=True)

env = Env(alpha=5, beta=6, mu=5, B=1000,
          c_r=-10, c_h=-1, gamma=-np.log(0.99),
          print_modulo=50)

self = {'name': 'Value Iteration',
        'v': np.zeros(env.B + 1, dtype=float),  # V_{t-1}
        'v_t': np.zeros(env.B + 1, dtype=float),  # V_t
        'pi': np.array([1]*env.B + [0], dtype=np.int32),
        'converged': False,
        'count': 0}
self = SimpleNamespace(**self)


def convergence(env, v_t, v, i, name, j=-1):
    """Convergence check."""
    diff = v_t - v
    delta_max = max(diff)
    delta_min = min(diff)
    converged = delta_max - delta_min < env.eps
    max_iter = (i > env.max_iter) | (j > env.max_iter)
    time = (clock() - env.start_time)
    max_time = time > env.max_time
    if (converged | (((i % env.print_modulo == 0) & (j == -1))
                     | (j % env.print_modulo == 0))):
        print("iter: ", i,
              "inner_iter: ", j,
              ", time: ", round(time, 2),
              ", delta: ", round(delta_max - delta_min, 2),
              ', D_min', round(delta_min, 2),
              ', D_max', round(delta_max, 2))
    if converged:
        if j == -1:
            print(name, 'converged in', i, 'iterations and ', time, 'seconds.')
        else:
            print(name, 'converged in', j, 'iterations and ', time, 'seconds.')
    elif max_iter:
        print(name, 'iter:', i, '(', j, ') reached max_iter =', max_iter,
              'after ', time, 'seconds.')
    elif max_time:
        print(name, 'iter:', i, '(', j, ') reached max_time =', max_time,
              'after ', time, 'seconds.')
    return converged, max_iter | max_time


def value_iteration(s, env):
    """Value iteration."""
    x = np.arange(env.B + 1)
    y_dep = [0] + list(range(env.B))
    stopped = False
    while not (stopped | s.converged):
        s.v_t[0] = env.lab * s.v[1]
        s.v_t[1:env.B] = env.lab * np.minimum(s.v[2:env.B+1],
                                              env.c_r + s.v[1:env.B])
        s.v_t[env.B] = env.lab * (env.c_r + s.v[env.B])
        s.v_t += env.c_h * x + env.mu * s.v[y_dep]
        s.v_t = s.v_t / (env.gamma + env.lab + env.mu)
        if s.count % env.convergence_check == 0:
            s.converged, stopped = convergence(env, s.v_t, s.v, s.count, s.name)
        s.v = s.v_t - s.v_t[0]  # Rescale V_t
        s.count += 1


if __name__ == "__main__":
    value_iteration(self, env)
    self.pi[1:env.B] = np.argmin([self.v[2:env.B+1],
                                  env.c_r + self.v[1:env.B]], axis=0)
    print(np.nonzero(self.pi == 0)[0][0])
    print(self.pi)
