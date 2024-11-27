import numpy as np
from utils import Env
from learners import FunctionApprox

class Sarsa:
    """Sarsa agent."""

    def __init__(self, env: Env, alpha=0.1, gamma=0.99, n=1, approx=False,
                 **kwargs):
        self.n = n  # n-step
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.approx = approx
        if approx is True:
            self.q_approx = FunctionApprox(momentum=kwargs.get('momentum', 0.9),
                                           gamma=self.gamma,
                                           n=self.n,
                                           method=kwargs.get('method', 'LSE'),
                                           control=True,
                                           clipping=kwargs.get('clipping',
                                                               False))
        else:
            self.q = np.zeros([env.B + 1, env.A])
        self.policies = Policies(method=kwargs.get('method', 'eps_greedy'),
                                 eps=kwargs.get('eps', 1e-2),
                                 eps_decay=kwargs.get('eps_decay', 0.95),
                                 c=kwargs.get('c', 1.0))
        self.a_t = self.choose(self, env)  # a_t
        self.a = None  # a_{t+1}

    def choose(self, env: Env):
        return self.policies.choose(env, self.q)

    def update_q(self, env: Env):
        """Update Q-value using sarsa."""
        if env.t - self.n + 1 < 0:
            return
        t = env.t - self.n
        g_t = env.n_step_return(self.gamma, self.n, t)
        target = g_t + self.gamma ^ self.n * self.q[env.x[env.t], env.a[env.t]]
        error = target - self.q[env.x[t], env.a[t]]
        self.q[env.x[t], env.a[t]] += self.alpha * error

    def learn(self, env: Env):
        """Update Q-value using sarsa."""
        if self.approx is True:
            self.q_approx.learn(env)
        else:
            self.update_q(env)