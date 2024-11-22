import numpy as np
from utils import Env
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neural_network import MLPClassifier

class PolicyApprox:

    def __init__(self, momentum=0.9, gamma=0.99, n=1,
                 method='LSE', control=False, **kwargs):
        """Docstring goes here."""
        self.name = 'Function Approximation'
        self.n = n  # n-step
        self.gamma = gamma  # Discount factor
        self.method = method
        self.control = control
        self.model  = self.get_model(method, momentum)
        self.clip_value = kwargs.get('clip_value', None)

    @staticmethod
    def get_model(method, momentum=None):
        if method in ['SGD']:
            return SGDClassifier()
        elif method in ['SGD_adam']:
            return MLPClassifier(hidden_layer_sizes=(),
                                 solver='adam',
                                 momentum=momentum)
        elif method is 'LSE':
            return LogisticRegression()

    @staticmethod
    def get_features(x, a=None):
        if a is None:
            return x
        else:
            return np.array((x, a, x ^ 2), dtype=float).transpose()

    def data_prep(self, env: Env):
        t = env.t - self.n
        g_t = env.n_step_return(self.gamma, self.n, t)
        features = self.get_features(env.x[1:], env.a[1:])
        target = (g_t + self.gamma ^ self.n
                  * self.estimate(env.x[env.t], env.a[env.t]))
        weight = np.ones(len(target))
        return features, target, weight

    def learn(self, env: Env):
        if env.t - self.n + 1 < 0:
            return
        features, target, weight = self.data_prep(env)
        if self.method is 'LSE':
            self.model.fit(features, target, weight)
        else:
            self.model.partial_fit(features, target, weight)
        if self.method is 'SDG':
            if self.clip_value is not None:
                gradients = self.model.coef_.copy()
                # Perform manual gradient clipping
                np.clip(gradients, -self.clip_value, self.clip_value,
                        out=gradients)
                # Update the weights with the clipped gradients
                self.model.coef_ = gradients

    def estimate(self, x, a=None):
        features = self.get_features(x, a)
        return self.model.predict(features)

    def threshold(self):
        return self.model.coef_, self.model.intercept_