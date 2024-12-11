import re
from utils import Env
import learners
import pickle

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
                                       eps=kwargs.get('eps', 1e-4))
    elif agent_name == 'certainty_equivalent':
        return learners.ValueIteration(env,
                                       method='certainty_equivalent',
                                       max_iter=kwargs.get('max_iter', 1e4),
                                       eps=kwargs.get('eps', 1e-4))
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

def run_simulation(env: Env, agent, steps):
    for step in range(int(steps)):
        # Choose an action
        action = agent.choose(env)
        # Take a step in the environment
        env.step(action)
        # Learn from experience
        agent.learn(env)
        # debug TODO
        agent.threshold(env, trace=True)

def multi_simulation(inst, inst_id, i, agent_name):
    env = Env(rho=inst.rho,
              alpha=inst.alpha,
              beta=inst.beta,
              B=inst.B,
              c_r=inst.c_r,
              c_h=inst.c_h,
              gamma=inst.gamma,
              steps=inst.steps,
              eps=inst.eps)
    memory = {'x': [], 'a': [], 'r': [], 'k': []}
    for episode in range(int(inst.episodes)):
        env.reset(seed=inst.seed + i * inst.episodes + episode)
        agent = agent_pick(env, agent_name)
        run_simulation(env, agent, inst.steps)
        # save simulation
        memory['x'].append(env.x)
        memory['a'].append(env.a)
        memory['r'].append(env.r)
        memory['k'].append(env.k)
    filename = (FILEPATH_DATA + '_' + inst_id + '_' + agent_name
                + '_' + str(i) + '.pickle')
    with open(filename, 'wb') as handle:
        pickle.dump(memory, handle)

# if isinstance(pi, int):  # Threshold value
#     a = (x < self.B) and (x < pi)  # admit if True
# else:  # policy vector
#     a = (x < self.B) and (pi[x] == 1)  # admit if True

# if 0 < env.x[-1] < env.B:
        #     action = agent.choose(env)
        # else:
        #     action = (env.x[-1] == 0)
