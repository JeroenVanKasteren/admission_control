"""
Creation of instance file for admission control problem.
ID of instance is the ID of the instance file combined with the index of the
instance in the file.
"""

import os
import pandas as pd
from sklearn.model_selection import ParameterGrid
from utils import agent_pick, run_simulation, Env
import scipy

def main(name):
    filepath = 'results/instances_' + name + '.csv'
    # Constants that are the same for all instances
    instance_columns = ['rho',  # load
                        'alpha',  # Shape param of gamma dist
                        'beta',  # Rate param of gamma dist
                        'B',  # Buffer size
                        'c_r',  # Rejection cost
                        'c_h',  # Holding cost
                        'gamma'  # Discount factor < 1
                        'steps',  # Number of steps per episode
                        'episodes',  # Number of episodes
                        'seed',  # seed for random number generator
                        'threshold_min',
                        'threshold',
                        'threshold_max']  # optimal threshold

    # Ensure method names are distinguishable (unique: '_' + method + '_job')
    # methods = ['full_info', 'certainty_equivalent', 'bdp',
    #            'sarsa_n1', 'sarsa_n10',
    #            'q_learning_n1', 'q_learning_n10',
    #            'reinforce', 'actor_critic']
    # instance_columns.extend(['N', 'start_K', 'batch_T'])
    # method_columns = ['_g', '_g_ci', '_perc', '_perc_ci']

    # state space bdp (D^J * S^J)
    # J = 1; D = 25*20; S = 5  # D = gamma * gamma_multi
    # print(f'{(J + 1) * D**J * S**J / 1e6} x e6')
    if name == 'lab_1':
        param_grid = {'rho': [0.6, 0.7, 0.8, 0.9],  # load
                      'alpha': [0.5, 1, 2, 2.5],  # Shape param of gamma dist
                      'beta': [1],  # Rate param of gamma dist
                      'B': [100],  # Buffer size
                      'c_r': [1],  # Rejection cost
                      'c_h': [1, 1.1, 1.2],  # Holding cost
                      'gamma': [0.9, 0.95, 0.99],  # Discount factor < 1
                      'steps': [1e4],
                      'episodes': [1e1],
                      'seed': [42]}
        grid = pd.DataFrame(ParameterGrid(param_grid))
        grid.beta = grid.alpha  # ensuring E(lambda) = 1
    elif name == 'high':  # Instance 3
        param_grid = {'rho': [0.9, 1, 1.5, 2],  # load
                      'alpha': [0.5, 1, 1.5],  # Shape param of gamma dist
                      'beta': [1],  # Rate param of gamma dist
                      'B': [100],  # Buffer size
                      'c_r': [1],  # Rejection cost
                      'c_h': [0.1, 0.5, 1],  # Holding cost
                      'gamma': [0.9, 0.95],  # Discount factor < 1
                      'steps': [1e4],
                      'episodes': [1e1],
                      'seed': [42]}
        grid = pd.DataFrame(ParameterGrid(param_grid))
        grid.beta = grid.alpha  # ensuring E(lambda) = 1
    else:
        print('Error: ID not recognized')
        exit(0)
    grid['eps'] = 1e-3  # epsilon
    grid['threshold_min'] = 0
    grid['threshold_mean'] = 0
    grid['threshold_max'] = 0

    for i, inst in grid.iterrows():
        inst = grid.iloc[i]
        env = Env(rho=inst.rho,
                  alpha=inst.alpha,
                  beta=inst.beta,
                  B=inst.B,
                  c_r=inst.c_r,
                  c_h=inst.c_h,
                  gamma=inst.gamma,
                  steps=inst.steps,
                  eps=inst.eps)
        lab_min = scipy.stats.gamma.isf(0.1, inst.alpha, scale=1 / inst.beta)
        lab_mean = inst.alpha / inst.beta
        lab_max = scipy.stats.gamma.isf(0.9, inst.alpha, scale=1 / inst.beta)
        labels = ['threshold_min', 'threshold_mean', 'threshold_max']
        for j, lab in enumerate([lab_min, lab_mean, lab_max]):
            env.reset(seed=inst.seed + i * inst.episodes, lab=lab)
            agent = agent_pick(env, 'full_info')
            grid.loc[i, labels[j]] = agent.threshold(env)

    # for method in methods:
    #     grid[method + '_iters'] = 0
    if os.path.isfile(filepath):
        print('Error: file already exists, name: ', filepath)
    else:
        grid.to_csv(filepath)

if __name__ == '__main__':
    for name in ['high']:
        main(name)
