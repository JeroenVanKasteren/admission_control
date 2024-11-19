import numpy as np
import pandas as pd
import re
from utils import tools, Env
import learners

# load in inst
instances_id = '1'
seed = 42
alpha = 1
beta = 1
B = 1  # Buffer size
c_r = 100  # Rejection cost
c_h = 1  # Holding cost
gamma = 0.9  # Discount factor < 1
eps = 1e-4
mu = 1

episodes = 10
steps = 1000
# max_time = '00:00:10'  # HH:MM:SS
# print_modulo = 10  # 1 for always
# convergence_check = 1e1
agents = ['full_info', 'certainty_equivalent', 'bdp',
          'sarsa_n1', 'sarsa_n10',
          'q_learning_n1', 'q_learning_n10',
          'actor_critic']

FILEPATH_INSTANCES = 'results/instances_' + instances_id + '.csv'
FILEPATH_DATA = 'results/data'

def agent_pick(env, name, **kwargs):
    # Extract n for n-step algorithms (number at the end of name after _n)
    match = re.search(name, r'_n(\d+)$')
    n = int(match.group(1)) if match else None
    # Remove '_n' followed by any number at the end of the string
    agent_name = re.sub(r'_n\d+$', '', name)
    if agent_name == 'full_info':
        return learners.ValueIteration(env,
                                       method='full_info',
                                       eps = kwargs.get('eps', 1e-2))
    elif agent_name == 'certainty_equivalent':
        return learners.ValueIteration(env,
                                       method='certainty_equivalent',
                                       eps = kwargs.get('eps', 1e-2))
    elif agent_name == 'bdp':
        return learners.BDP(env)
    elif agent_name == 'sarsa':
        return learners.Sarsa(env, n)
    elif agent_name == 'q_learning':
        return learners.QLearning(env, n)
    elif agent_name == 'actor_critic':
        return learners.ActorCritic(env)

def train(env: Env, agent, memory):
    for step in range(steps):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            # Choose an action
            action = agent.choose(env)

            # Take a step in the environment
            next_state, reward, done = env.step(action)

            # Learn from experience
            agent.learn(state, action, reward, next_state)
        memory['x'].append(env.x)
        memory['a'].append(env.a)
        memory['r'].append(env.r)
        memory['k'].append(env.k)
    return memory

insts = pd.read_csv(FILEPATH_INSTANCES)
for i, inst in insts.iterrows():
    env = Env(seed=seed, alpha=alpha, beta=beta, B=B, c_r=c_r, c_h=c_h,
              gamma=gamma, eps=eps, mu=mu, max_iter=steps)
    for agent_name in agents:
        memory = {'x': [], 'a': [], 'r': [], 'k': []}
        for episode in range(episodes):
            agent = agent_pick(env, agent_name)
            memory = train(env, agent, memory)
        np.savez(FILEPATH_DATA + instances_id + '_' + str(i)
                 + '_' + agent_name + '.npz', memory)
