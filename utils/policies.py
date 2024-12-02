"""
General policies for reinforcement learning.
"""

import numpy as np
from utils.env import Env

class Policies:
    """..."""

    def __init__(self, method: str, **kwargs):
        self.method = method
        if method == 'eps_greedy':
            self.eps = kwargs.get('eps', 1e-2)  # Exploration rate
            self.eps_decay = kwargs.get('eps_decay', 0.95)  # Decay rate
        elif method == 'ucb':
            self.c = kwargs.get('c', 1.0)  # Exploration factor


    def choose(self, env: Env, q: np.ndarray):
        """Choose an action."""
        if self.method == 'greedy':
            return np.argmax(q[env.x, :])
        elif self.method == 'eps_greedy':
            return self.eps_greedy(env, q)
        elif self.method == 'ucb':
            return self.ucb(env, q)
        else:
            raise ValueError('Invalid method')

    def eps_greedy(self, env: Env, q: np.ndarray):
        """Choose an action based on the epsilon-greedy policy."""
        self.eps *= self.eps_decay
        if env.rng.uniform(0, 1) < self.eps:
            # Exploration: choose a random action
            a = env.rng.choice(range(env.A))
        else:
            # Exploitation: choose the best action based on Q value
            a = np.argmax(q[env.x, :])
        self.eps *= self.eps_decay
        return a

    # TODO, check generated code!
    def ucb(self, env: Env, q: np.ndarray):
        """Choose an action based on the upper confidence bound (UCB) policy."""
        n = np.sum(q[:, 1])
        ucb = q[:, 0] + self.c * np.sqrt(np.log(n) / q[:, 1])
        return np.argmax(ucb)