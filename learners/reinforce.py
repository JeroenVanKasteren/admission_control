import numpy as np
from utils import Env
from leaners import FunctionApprox

class Reinforce:

    def __init__(self, momentum=0.9, gamma=0.99, n=1,
                 method='LSE', control=False, **kwargs):
        """Docstring goes here."""
        self.name = 'Reinforce'
        self.n = n  # n-step
        self.gamma = gamma  # Discount factor
        self.method = method
        self.control = control
        self.baseline = FunctionApprox()
        self.model  = FunctionApprox.get_model(method, momentum)
        self.clip_value = kwargs.get('clip_value', None)
