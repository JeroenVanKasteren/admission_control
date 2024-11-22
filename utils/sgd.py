import numpy as np

class SGDLinear:
    def __init__(self, learning_rate=0.01, momentum=0.9, n_features=1):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.coef_ = np.ones(n_features)
        self.change =

    def predict(self, features):
        """Return f(x) = w^T x."""
        return np.dot(self.coef_, features)

    @staticmethod
    def gradient(features):
        """Return the gradient of f(x) = w^T x w.r.t. w."""
        return features

    def partial_fit(self, features, target, weight):
        gradient = features  #
        error = target - self.predict(features)
        self.coef_ += self.learning_rate * error * gradient


