"""
Creation of instance file for admission control problem.
ID of instance is the ID of the instance file combined with the index of the
instance in the file.
"""

import os
import pandas as pd
from sklearn.model_selection import ParameterGrid

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
                        'seed']  # seed for random number generator

    # Ensure method names are distinguishable (unique: '_' + method + '_job')
    methods = ['full_info', 'certainty_equivalent', 'bdp',
               'sarsa_n1', 'sarsa_n10',
               'q_learning_n1', 'q_learning_n10',
               'reinforce', 'actor_critic']
    # instance_columns.extend(['N', 'start_K', 'batch_T'])
    # method_columns = ['_g', '_g_ci', '_perc', '_perc_ci']

    # for method in methods:
    #     instance_columns.extend([method + s for s in method_columns])
    #     if solve and method != 'vi':
    #         instance_columns.extend([method + s for s in heuristic_columns])

    # state space bdp (D^J * S^J)
    # J = 1; D = 25*20; S = 5  # D = gamma * gamma_multi
    # print(f'{(J + 1) * D**J * S**J / 1e6} x e6')
    if name == 'lab_1':
        param_grid = {'rho': [0.6, 0.7, 0.8, 0.9],  # load
                      'alpha': [0.5, 1, 2, 2.5],  # Shape param of gamma dist
                      'beta': [1],  # Shape param of gamma dist
                      'B': [100],  # Buffer size
                      'c_r': [1],  # Rejection cost
                      'c_h': [1 / 10, 2 / 10],  # Holding cost
                      'gamma': [0.8, 0.9, 0.95, 0.99],
                      'steps': [1e4],
                      'episodes': [1e1],
                      'seed': [42]}  # Discount factor < 1
        grid = pd.DataFrame(ParameterGrid(param_grid))
        grid.beta = grid.alpha
    elif name == 'J3':  # Instance 3
        exit(0)
    else:
        print('Error: ID not recognized')
        exit(0)
    grid['eps'] = 1e-3  # epsilon

    for method in methods:
        grid[method + '_iters'] = 0
    if os.path.isfile(filepath):
        print('Error: file already exists, name: ', filepath)
    else:
        grid.to_csv(filepath)

if __name__ == '__main__':
    for name in ['lab_1']:
        main(name)
