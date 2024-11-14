import numpy as np
from utils.env import Env

class QLearning:
    """Q-learning agent."""

    def __init__(self, env, alpha=0.1,
                 gamma=0.99, eps=1.0, eps_decay=0.995):
        # Q-table, initialized to zero, may switch to defaultdict
        self.q = np.zeros(env.state_size, env.action_size)

        # Hyperparameters
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.eps = eps  # Exploration rate
        self.eps_decay = eps_decay  # Decay rate for exploration

    def eps_greedy(self, env):
        """Choose an action based on the epsilon-greedy policy."""
        if env.rng.uniform(0, 1) < self.eps:
            # Exploration: choose a random action
            return env.rng.choice(range(env.action_size))
        else:
            # Exploitation: choose the best action based on Q value
            return np.argmax(self.q[env.x, :])

    def update(self):
        """Update Q-value using the Q-learning formula."""
        a_n = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * \
                    self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error

    def decay_epsilon(self):
        """Decay the exploration rate."""
        self.eps = self.eps * self.eps_decay

    def act(self, env):
        """Choose an action."""
        self.update(env)
        return self.eps_greedy(env)
