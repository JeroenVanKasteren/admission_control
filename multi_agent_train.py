import numpy as np
import pandas as pd
import re
from utils import tools, Env
import learners

# load in inst
instances_id = 'lab_1'
seed = 42

# max_time = '00:00:10'  # HH:MM:SS
# print_modulo = 10  # 1 for always
# convergence_check = 1e1
agents = ['full_info']  # , 'certainty_equivalent', 'bdp']
# agents = ['full_info', 'certainty_equivalent', 'bdp',
#           'sarsa_n1', 'sarsa_n10',
#           'q_learning_n1', 'q_learning_n10',
#           'reinforce', 'actor_critic']

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
                                       max_iter=kwargs.get('max_iter', 1e4),
                                       eps = kwargs.get('eps', 1e-4))
    elif agent_name == 'certainty_equivalent':
        return learners.ValueIteration(env,
                                       method='certainty_equivalent',
                                       max_iter=kwargs.get('max_iter', 1e4),
                                       eps = kwargs.get('eps', 1e-4))
    elif agent_name == 'bdp':
        return learners.BDP(env)
    elif agent_name == 'sarsa':
        return learners.Sarsa(env, n)
    elif agent_name == 'q_learning':
        return learners.QLearning(env, n)
    # elif agent_name == 'reinforce':
    #     return learners.Reinforce(env)
    # elif agent_name == 'actor_critic':
    #     return learners.ActorCritic(env)

def train(env: Env, agent, memory, steps):
    for step in range(int(steps)):
        state = env.reset()
        done = False
        while not done:
            # Choose an action
            action = agent.choose(env)
            # Take a step in the environment
            env.step(action)
            # Learn from experience
            agent.learn(env)
        memory['x'].append(env.x)
        memory['a'].append(env.a)
        memory['r'].append(env.r)
        memory['k'].append(env.k)
    return memory

instances = pd.read_csv(FILEPATH_INSTANCES)
for i, inst in instances.iterrows():
    env = Env(rho=inst.rho,
              alpha=inst.alpha,
              beta=inst.beta,
              B=inst.B,
              c_r=inst.c_r,
              c_h=inst.c_h,
              gamma=inst.gamma,
              steps=inst.steps,
              seed=inst.seed + i,
              eps=inst.eps)
    for agent_name in agents:
        memory = {'x': [], 'a': [], 'r': [], 'k': []}
        for episode in range(int(inst.episodes)):
            agent = agent_pick(env, agent_name)
            memory = train(env, agent, memory, inst.steps)
        np.savez(FILEPATH_DATA + instances_id + '_' + agent_name
                 + '_' + str(i) + '.npz', memory)

# if isinstance(pi, int):  # Threshold value
#     a = (x < self.B) and (x < pi)  # admit if True
# else:  # policy vector
#     a = (x < self.B) and (pi[x] == 1)  # admit if True