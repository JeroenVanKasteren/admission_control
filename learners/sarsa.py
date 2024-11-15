import numpy as np
from utils.env import Env

class QLearning:
    """Q-learning agent."""

    def __init__(self, env: Env, alpha=0.1, gamma=0.99, n=1):
        # Q-table, initialized to zero, may switch to defaultdict
        self.q = np.zeros([env.B + 1, env.A])
        self.n = n  # n-step
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor

    def learn(self, env: Env):
        """Update Q-value using sarsa."""
        if env.t < self.n:
            return
        t = env.t - self.n
        g_t = env.n_step_return(self.gamma, self.n, t)
        target = g_t + self.gamma ^ self.n * self.q[env.x[env.t], env.a[env.t]]
        error = target - self.q[env.x[t], env.a[t]]
        self.q[env.x[t], env.a[t]] += self.alpha * error

