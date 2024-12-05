import numpy as np
import matplotlib as plt
import pandas as pd
from utils import multi_boxplot

instances_id = 'lab_1'

FILEPATH_INSTANCES = 'results/instances_' + instances_id + '.csv'
FILEPATH_DATA = 'results/data'

colors = iter([plt.cm.tab20(i) for i in range(20)])
alpha = .2
agents = {'full_info': [next(colors), {}],
          'certainty_equivalent': [next(colors), {}],
          'bdp': [next(colors), {}],
          'sarsa_n1': [next(colors), {}],
          'sarsa_n10': [next(colors), {}],
          'q_learning_n1': [next(colors), {}],
          'q_learning_n10': [next(colors), {}],
          'reinforce': [next(colors), {}],
          'actor_critic': [next(colors), {}]}
agent_ref = 'full_info'

instances = pd.read_csv(FILEPATH_INSTANCES)

i = 0
boxplot_view = 'mean'  # mean, median, all
inst = instances.iloc[i]
steps = inst.steps
episodes = inst.episodes
x = np.arange(steps)
steps_measure = [10 ** (p + 1) for p in range(1, int(np.log10(steps)))]
memory_ref = np.load(FILEPATH_DATA  + instances_id + '_' + agent_ref
                     + '_' + str(i) + '.npz')['arr_0']

for agent in agents:
    for step in steps_measure:
        agents[agent][1][step] = []

r_ref = np.cumsum(memory_ref.r, axis=1)  # cumsum per episode (over steps)
for agent in agents:
    # load instance
    memory = np.load(FILEPATH_DATA  + instances_id + '_' + agent
                     + '_' + str(i) + '.npz')['arr_0']
    regret = np.cumsum(memory.r, axis=1) - r_ref
    for step in steps_measure:
        if boxplot_view == 'mean':
            agents[agent][1][step].append(np.mean(regret[:, step])/step)
        elif boxplot_view == 'median':
            agents[agent][1][step].append(np.median(regret[:, step])/step)
        else:  # all
            agents[agent][1][step].extend(regret[:, step]/step)
    plt.plot(x, np.mean(regret, axis=0), lw=2)  # mean per step (over episodes)
    plt.fill_between(x,
                     np.max(regret, axis=0),
                     np.min(regret, axis=0),
                     color=agents[agent], alpha=alpha)
plt.show()

for agent in agents:
    multi_boxplot(agents[agent][1],
                  steps_measure,
                  'Regret of ' + agent,
                  steps_measure,
                  'Average regret per step',
                  log_y = True)
