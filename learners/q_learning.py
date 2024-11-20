import numpy as np
from utils.env import Env
from utils.policies import Policies

class QLearning:
    """Q-learning agent."""

    def __init__(self, env, alpha=0.1, gamma=0.99, n=1,
                 method='eps_greedy', **kwargs):
        self.q = np.zeros(env.state_size, env.action_size)
        self.n = n  # n-step
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.policies = Policies(method=method,
                                 eps=kwargs.get('eps', 1e-2),
                                 eps_decay=kwargs.get('eps_decay', 0.95),
                                 c=kwargs.get('c', 1.0),)

    def choose(self, env: Env):
        """Choose an action."""
        return np.argmax(self.q[env.x, :])

    def learn(self, env: Env):
        """Update Q-value using Q-learning."""
        if env.t - self.n + 1 < 0:
            return
        t = env.t - self.n
        g_t = env.n_step_return(self.gamma, self.n, t)
        target = g_t + self.gamma ^ self.n * max(self.q[env.x[env.t], :])
        error = target - self.q[env.x[t], env.a[t]]
        self.q[env.x[t], env.a[t]] += self.alpha * error
