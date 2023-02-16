"""
Sandbox Value Iteration
"""

import numpy as np
from numpy import array, arange
from src.Env import Env
from types import SimpleNamespace

np.set_printoptions(precision=4, linewidth=150, suppress=True)

env = Env(alpha=5, beta=6, mu=5, B=100,
          c_r=-10, c_h=-1, gamma=-np.log(0.99),
          print_modulo=10)

self = {'name': 'Value Iteration',
        'v': np.zeros(env.B + 1, dtype=float),  # V_{t-1}}
        'pi': np.append(np.ones(env.B, dtype=float), 0),
        'g': 0.0,
        'count': 0}
self = SimpleNamespace(**self)


def convergence(env, v_t, v, i, name, j=-1):
    """Convergence check of valid states only."""
    diff = v_t - v
    delta_max = max(diff)
    delta_min = min(diff)
    converged = delta_max - delta_min < env.eps
    max_iter = (i > env.max_iter) | (j > env.max_iter)
    g = (delta_max + delta_min) / 2 * (env.lab + env.mu)
    if ((converged & env.trace) |
            (env.trace & ((i % env.print_modulo == 0) |
                          (j % env.print_modulo == 0)))):
        print("iter: ", i, ", ", j,
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


def calculate_v(env, v):
    """Calculate V_{t+1}."""
    y_dep = [0] + list(range(env.B))
    y_arr = list(range(1, env.B + 1)) + [env.B]
    c_r_v = array([0] * env.B + [env.c_r])
    v_t = (env.c_h * arange(env.B + 1)
           + env.lab * np.minimum(v[y_arr] + c_r_v, env.c_r + v)
           + env.mu * v[y_dep])
    return v_t / (env.gamma + env.lab + env.mu)


def value_iteration(s, env):
    """Docstring."""
    start = env.timer(name=s.name)
    while True:  # Update each state.
        v_t = calculate_v(env, s.v)
        converged, s.g = convergence(env, v_t, s.v, s.count, s.name)
        if converged:
            break  # Stopping condition
        s.v = v_t - v_t[0]  # Rescale and Save v_t
        s.count += 1
    env.timer(name=s.name, start=start)


value_iteration(self, env)
self.v = self.v - self.v[0]

self.pi[1:env.B] = np.argmin([self.v[2:env.B+1],
                              env.c_r + self.v[1:env.B]], 0)
print(np.nonzero(self.pi == 0)[0][0])
print(self.pi)
