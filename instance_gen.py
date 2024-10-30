"""
Creation of instance file for admission control problem.
ID of instance is the ID of the instance file combined with the index of the
instance in the file.
"""

import numpy as np
import os
from utils import tools
import pandas as pd

def main(name, solve):
    filepath = 'results/instances_' + name
    # Constants that are the same for all instances
    ...
    instance_columns = ['alpha',
                        'beta',
                        'B',  # Buffer size
                        'c_r',  # Rejection cost
                        'c_h',  # Holding cost
                        'gamma'  # Discount factor < 1
                        'eps']  # epsilon

    # Ensure method names are distinguishable (unique: '_' + method + '_job')
    methods = ['average', 'bdp', 'q_learning', 'actor_critic']
    # instance_columns.extend(['N', 'start_K', 'batch_T'])
    # method_columns = ['_g', '_g_ci', '_perc', '_perc_ci']

    # for method in methods:
    #     instance_columns.extend([method + s for s in method_columns])
    #     if solve and method != 'vi':
    #         instance_columns.extend([method + s for s in heuristic_columns])

    # state space (J * D^J * S^J)
    # J = 1; D = 25*20; S = 5  # D = gamma * gamma_multi
    # print(f'{(J + 1) * D**J * S**J / 1e6} x e6')
    if name == '1':
        param_grid = {'alpha' = [1],
                      'beta' = [1],
        'B' = [100],  # Buffer size
                        'c_r' = [1],  # Rejection cost
                        'c_h' = [1/10],  # Holding cost
                        'gamma' = [10],  # Discount factor < 1
                        'eps' = [1e-4]}  # epsilon
    elif name == 'J3':  # Instance 3
        mu = 4
        param_grid = {'J': [3],
                      'S': [2, 3],
                      'D': [0, -4],
                      'gamma': [10],
                      'mu': [[mu, 1.5*mu, 2*mu]],
                      'load': [0.7, 0.9],
                      'imbalance': [[1/3, 2/3, 1], [1, 1, 1]]}
    elif name == 'J2_D_gam':  # Instance 2  # gamma multi = 8
        mu = 4
        param_grid = {'J': [2],
                      'S': [2, 5],
                      'D': [0, -5, -10],
                      'gamma': [10, 15, 20],
                      'mu': [[mu, mu], [mu, 2*mu]],
                      'load': [0.7, 0.9],
                      'imbalance': [[1/3, 1], [1, 1], [3, 1]]}
    elif name == 'J1':
        mu = 2
        param_grid = {'J': [1],
                      'S': [2, 4, 6],
                      'D': [0],
                      'gamma': [10, 15, 20, 25],
                      'mu': [[mu], [2*mu], [3*mu]],
                      'load': [0.7, 0.8, 0.9],
                      'imbalance': [[1]]}
    elif name == 'sim':
        grid = pd.DataFrame()
        for sim_id in sim_ids:
            param_grid = instances_sim.generate_instance(sim_id)
            for key, value in param_grid.items():
                param_grid[key] = [value]  # put in lists for ParameterGrid
            row = tools.get_instance_grid(param_grid, sim=True)
            grid = pd.concat([grid, row], ignore_index=True)
    else:
        print('Error: ID not recognized')
        exit(0)

    if name != 'sim':
        grid = tools.get_instance_grid(param_grid, max_t_prob=max_t_prob,
                                       max_size=max_size,
                                       del_t_prob=del_t_prob,
                                       del_size=del_size)
    if solve:
        grid['e'] = epsilon
        grid['P'] = P
        # Derive solved from g value.
        for method in methods:
            grid[method + '_job_id'] = ''
            grid[method + '_attempts'] = 0
            grid[method + '_time'] = '00:00'
            grid[method + '_iter'] = 0
            grid[method + '_g_tmp'] = np.nan
            grid[method + '_g'] = np.nan
            if method != 'vi':
                grid[method + '_opt_gap_tmp'] = np.nan
                grid[method + '_opt_gap'] = np.nan
    else:
        grid[['N', 'start_K', 'batch_T']] = 0
        for method in methods:
            grid[method + '_job_id'] = ''
            grid[method + '_attempts'] = 0
            grid[method + '_time'] = '00:00'
            grid[method + '_iter'] = 0
            grid[method + '_g'] = np.nan
            grid[method + '_g_ci'] = np.nan
            grid[method + '_perc'] = np.nan
            grid[method + '_perc_ci'] = np.nan

    grid = grid[instance_columns]

    if os.path.isfile(filepath):
        print('Error: file already exists, name: ', filepath)
    else:
        grid.to_csv(filepath)


if __name__ == '__main__':
    for name in ['J1', 'J2', 'J2_D_gam']:
        main(name, True)
    # for name in ['J1', 'J2', 'sim']:
    #     main(name, False)
    # main('J3', False)
