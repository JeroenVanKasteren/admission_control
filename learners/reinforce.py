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

# https://github.com/pytorch/examples/blob/main/reinforcement_learning/reinforce.py
# https://medium.com/@sofeikov/reinforce-algorithm-reinforcement-learning-from-scratch-in-pytorch-41fcccafa107

# policy_net = PolicyNet()
# policy_net.to(device)

#lengths = []
#rewards = []

gamma = 0.99
alpha = 2**-13
optimizer = torch.optim.Adam(policy_net.parameters(), lr=alpha)

prefix = "reinforce-per-step"

for episode_num in tqdm(range(2500)):
    all_iterations = []
    all_log_probs = []
    grid = get_good_starting_grid()
    episode = list(generate_episode(grid, policy_net=policy_net, device=device))
    lengths.append(len(episode))
    loss = 0
    for t, ((state, action, reward), log_probs) in enumerate(episode[:-1]):
        gammas_vec = gamma ** (torch.arange(t+1, len(episode))-t-1)
        # Since the reward is -1 for all steps except the last, we can just sum the gammas
        G = - torch.sum(gammas_vec)
        rewards.append(G.item())
        policy_loss = log_probs[action]
        optimizer.zero_grad()
        gradients_wrt_params(policy_net, policy_loss)
        update_params(policy_net, alpha * G * gamma**t)
