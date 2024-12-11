import pandas as pd
from utils import multi_simulation

# load in inst
instances_id = 'small'
seed = 42

FILEPATH_INSTANCES = 'results/instances_' + instances_id + '.csv'
FILEPATH_DATA = 'results/data'

# max_time = '00:00:10'  # HH:MM:SS
# print_modulo = 10  # 1 for always
# convergence_check = 1e1
agents = ['full_info', 'certainty_equivalent', 'bdp']
# agents = ['certainty_equivalent']
# agents = ['full_info', 'certainty_equivalent', 'bdp',
#           'sarsa_n1', 'sarsa_n10',
#           'q_learning_n1', 'q_learning_n10',
#           'reinforce', 'actor_critic']

instances = pd.read_csv(FILEPATH_INSTANCES)
i = 0  # Debug
inst = instances.iloc[i]  # Debug
# for i, inst in instances.iterrows():
for agent_name in agents:
    multi_simulation(inst, instances_id, i, agent_name)
