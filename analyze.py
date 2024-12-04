import numpy as np
import matplotlib as plt
import pandas as pd

instances_id = 'lab_1'

FILEPATH_INSTANCES = 'results/instances_' + instances_id + '.csv'
FILEPATH_DATA = 'results/data'

colors = iter([plt.cm.tab20(i) for i in range(20)])
alpha = .2
agents = {'full_info': [next(colors)],
          'certainty_equivalent': [next(colors)],
          'bdp': [next(colors)],
          'sarsa_n1': [next(colors)],
          'sarsa_n10': [next(colors)],
          'q_learning_n1': [next(colors)],
          'q_learning_n10': [next(colors)],
          'reinforce': [next(colors)],
          'actor_critic': [next(colors)]}
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

r_ref = np.cumsum(memory_ref.r, axis=1)  # cumsum per episode (over steps)
for agent in agents.keys():
    # load instance
    memory = np.load(FILEPATH_DATA  + instances_id + '_' + agent
                     + '_' + str(i) + '.npz')['arr_0']
    regret = np.cumsum(memory.r, axis=1) - r_ref
    for step in steps_measure:
        if boxplot_view == 'mean':
            agents[agent].append(np.mean(regret[:, step]))
        elif boxplot_view == 'median':
            agents[agent].append(np.median(regret[:, step]))
        else:  # all
            agents[agent].extend(regret[:, step])
    plt.plot(x, np.mean(regret, axis=0), lw=2)  # mean per step (over episodes)
    plt.fill_between(x,
                     np.max(regret, axis=0),
                     np.min(regret, axis=0),
                     color=agents[agent], alpha=alpha)
plt.show()

# plot
