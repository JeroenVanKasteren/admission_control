import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from utils import multi_boxplot

instances_id = 'small'

FILEPATH_INSTANCES = 'results/instances_' + instances_id + '.csv'
FILEPATH_DATA = 'results/data'

colors = iter([plt.cm.tab20(i) for i in range(20)])
alpha = .2
agents = {'full_info': [next(colors), {}],
          'certainty_equivalent': [next(colors), {}]}
          # 'bdp': [next(colors), {}],
          # 'sarsa_n1': [next(colors), {}],
          # 'sarsa_n10': [next(colors), {}],
          # 'q_learning_n1': [next(colors), {}],
          # 'q_learning_n10': [next(colors), {}],
          # 'reinforce': [next(colors), {}],
          # 'actor_critic': [next(colors), {}]}
agent_ref = 'full_info'

instances = pd.read_csv(FILEPATH_INSTANCES)

i = 0
boxplot_view = 'mean'  # mean, median, all
inst = instances.iloc[i]
steps = inst.steps
episodes = inst.episodes
steps_measure = [10 ** (p + 1) for p in range(1, int(np.log10(steps)))]
filename = (FILEPATH_DATA + '_' + instances_id + '_' + agent_ref
            + '_' + str(i) + '.pickle')
with open(filename, 'rb') as handle:
    memory_ref = pickle.load(handle)

for agent in agents:
    for step in steps_measure:
        agents[agent][1][step] = []

r_ref = np.array(list(itertools.zip_longest(*memory_ref['r'],
                                            fillvalue=np.nan))).T
r_ref = np.cumsum(r_ref, axis=1)  # cumsum per episode (over steps)
# agent = 'certainty_equivalent'
for agent in agents:
    # load instance
    filename = (FILEPATH_DATA + '_' + instances_id + '_' + agent
                + '_' + str(i) + '.pickle')
    with open(filename, 'rb') as handle:
        memory = pickle.load(handle)
    r_agent = np.array(list(itertools.zip_longest(*memory['r'],
                                                  fillvalue=np.nan))).T
    r_agent = np.cumsum(r_agent, axis=1)  # cumsum per episode (over steps)
    # regret per episode (over steps)
    regret = r_agent - r_ref
    for step in steps_measure:
        if boxplot_view == 'mean':
            agents[agent][1][step].append(np.mean(regret[:, step])/step)
        elif boxplot_view == 'median':
            agents[agent][1][step].append(np.median(regret[:, step])/step)
        else:  # all
            agents[agent][1][step].extend(regret[:, step]/step)
    x = np.arange(len(regret[0]))
    # mean per step (over episodes)
    plt.plot(x, np.mean(regret, axis=0), lw=2, label=agent)
    plt.fill_between(x,
                     np.max(regret, axis=0),
                     np.min(regret, axis=0),
                     color=agents[agent][0], alpha=alpha)
plt.title('Regret')
plt.ylabel('Regret')
plt.xlabel('events')
plt.legend()
plt.show()

for agent in agents:
    multi_boxplot(agents[agent][1],
                  steps_measure,
                  'Regret of ' + agent,
                  steps_measure,
                  'Average regret per step',
                  x_label='Events')
# ,log_y = True)
