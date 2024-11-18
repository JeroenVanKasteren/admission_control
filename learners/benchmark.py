import numpy as np
from utils.env import Env

class Benchmark:
    """Benchmark agent."""

    def __init__(self, env: Env, method='dynamic'):
        self.method = method
        if self.method is 'full_info':
            self.lab = env.lab
        else:
            self.lab = env.alpha / env.beta

    def learn(self, env: Env):
        if self.method is 'dynamic':
            self.lab = env.k / env.t

    def choose(self, env: Env):
        return env.x[env.t] * env.c_h > self.lab * env.c_r
