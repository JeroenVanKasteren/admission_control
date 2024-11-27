import numpy as np
from utils import Env
from sklearn.linear_model import LinearRegression
import torch

class FunctionApprox:

    def __init__(self, momentum=0.9, gamma=0.99, n=1,
                 method='LSE', control=False, **kwargs):
        """Docstring goes here."""
        self.name = 'Function Approximation'
        self.n = n  # n-step
        self.gamma = gamma  # Discount factor
        self.method = method

        self.control = control
        self.input_dim = kwargs.get('input_dim', 1 + control)
        self.output_dim = kwargs.get('output_dim', 1)
        self.lr = kwargs.get('lr', 1e-2)  # learning_rate

        self.model, self.criterion, self.optimizer  = self.get_model(method, momentum)
        self.clip_value = kwargs.get('clip_value', None)

    def get_model(self, method, momentum=None):
        if method in ['SGD', 'SGD_adam']:
            model = torch.nn.linear(in_features=self.input_dim,
                                    out_features=1)
            criterion = torch.nn.MSELoss()
            if method == 'SGD':
                optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
            else:
                optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        elif method == 'LSE':
            model = LinearRegression()
            criterion, optimizer = None, None
        else:
            raise ValueError('Invalid method')
        return model, criterion, optimizer

    def get_features(self, x, a=None):
        if self.control:
            if self.input_dim == 2:
                return [x, a]
            elif self.input_dim == 3:
                return [x, a, x ^ 2]
            else:
                return [x, a, x ^ 2, x / (x - 1)]
        else:
            if self.input_dim == 1:
                return [x]
            elif self.input_dim == 2:
                return [x, x ^ 2]
            else:
                return [x, x ^ 2, x / (x - 1)]

    def learn(self, env: Env):
        t = env.t - self.n
        if t < 0:
            return
        g_t = env.n_step_return(self.gamma, self.n, t)
        features = self.get_features(env.x[1:], env.a[1:])
        if self.method == 'LSE':
            target = (g_t + self.gamma ^ self.n
                      * self.model.predict(features[-1]))
            self.model.fit(features, target)
        else:
            target = (g_t + self.gamma ^ self.n
                      * self.model.predict(features[-1]))
            y_hat = self.model(torch.FloatTensor(features))
            loss = self.criterion(y_hat, torch.FloatTensor(target))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
