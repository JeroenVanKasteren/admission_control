from utils.env import Env
from learners import QLearning, BDP, ActorCritic, ValueIteration, PolicyIteration
# Init

seed = 42
alpha = 1
beta = 1
B = 1  # Buffer size
c_r = 100  # Rejection cost
c_h = 1  # Holding cost
gamma = 0.9  # Discount factor < 1
eps = 1e-4
mu = 1

max_iter = 1e6
# max_time = '00:00:10'  # HH:MM:SS
# print_modulo = 10  # 1 for always
# convergence_check = 1e1

def train(episodes=1000):
    env = Env(seed=seed, alpha=alpha, beta=beta, B=B, c_r=c_r, c_h=c_h,
              gamma=gamma, eps=eps, mu=mu, max_iter=max_iter)
    agent = QLearning(state_size=env.state_space,
                           action_size=env.action_space)

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            # Choose an action
            action = agent.choose_action(state)

            # Take a step in the environment
            next_state, reward, done = env.step(action)

            # Update the Q-value
            agent.update_q_value(state, action, reward, next_state)

            # Move to the next state
            state = next_state
            total_reward += reward

        # Decay the exploration rate
        agent.decay_epsilon()

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}: Total Reward: {total_reward}")

    return agent