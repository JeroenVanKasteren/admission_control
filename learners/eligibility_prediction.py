import numpy as np
from utils.env import Env
from utils.policies import Policies

class EligibilityPrediction:
    """(True) on-line TD(lambda), Eligibility Traces agent."""

    def __init__(self, env, alpha=0.1, gamma=0.99, n=1,
                 method='true_online', **kwargs):
        self.v = np.zeros(env.state_size)
        if method == 'true_online':
            self.v_old = 0
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        pi = kwargs.get('pi')
        if isinstance(pi, (float, int)):
            self.pi = np.zeros(pi)
            self.pi[:pi] = 0
        else:
            self.pi = pi

    def choose(self, env: Env):
        """Choose an action."""
        return self.policies.choose(env, self.q)

    def learn(self, env: Env):
        """Update Q-value using Q-learning."""
        if env.t - self.n + 1 < 0:
            return
        t = env.t - self.n
        g_t = env.n_step_return(self.gamma, self.n, t)
        target = g_t + self.gamma ^ self.n * max(self.q[env.x[env.t], :])
        error = target - self.q[env.x[t], env.a[t]]
        self.q[env.x[t], env.a[t]] += self.alpha * error
