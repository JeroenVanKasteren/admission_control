import pandas as pd
from utils import run_simulation, simulations
# load in inst
instances_id = 'lab_1'
seed = 42

FILEPATH_INSTANCES = 'results/instances_' + instances_id + '.csv'
FILEPATH_DATA = 'results/data'

# max_time = '00:00:10'  # HH:MM:SS
# print_modulo = 10  # 1 for always
# convergence_check = 1e1
agents = ['certainty_equivalent']
# agents = ['full_info', 'certainty_equivalent', 'bdp',
#           'sarsa_n1', 'sarsa_n10',
#           'q_learning_n1', 'q_learning_n10',
#           'reinforce', 'actor_critic']

instances = pd.read_csv(FILEPATH_INSTANCES)
# for i, inst in instances.iterrows():
i = 0
inst = instances.iloc[i]
for agent_name in agents:
    simulations(inst, agent_name, episodes=inst.episodes)
