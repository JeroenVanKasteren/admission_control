import numpy as np
from utils import Env
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.neural_network import MLPRegressor

class FunctionApprox:

    def __init__(self, env: Env, momentum=0.9, gamma=0.99, n=1,
                 method='LSE', **kwargs):
        """Docstring goes here."""
        self.name = 'Function Approximation'
        self.n = n  # n-step
        self.gamma = gamma  # Discount factor
        self.method = method
        if self.method in ['SGD']:
            self.model = SGDRegressor()
        elif self.method in ['SGD_adam']:
            self.model = MLPRegressor(hidden_layer_sizes=(),
                                      solver='adam',
                                      momentum=momentum)
        elif self.method == 'LSE':
            self.model = LinearRegression()
        if kwargs.get('clipping', False):
            self.clip_value = kwargs.get('clip_value', 1.0)
        else:
            self.clip_value = None

    def learn(self, env: Env):
        if self.method == 'SDG':
            # sample_weight = ...
            self.model.partial_fit(env.x[1:], env.r[1:])  # , sample_weight)
            if self.clip_value is not None:
                gradients = self.model.coef_.copy()
                # Perform manual gradient clipping
                np.clip(gradients, -self.clip_value, self.clip_value,
                        out=gradients)
                # Update the weights with the clipped gradients
                self.model.coef_ = gradients
        else:
            # sample_weight = ...
            self.model.fit(env.x[1:], env.r[1:])  # , sample_weight)
    def estimate(self, x):
        return self.model.predict(x)

