import gym
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Directory to which qtables will be saved
DIR_Q_TABLES = 'qtables'
Path(DIR_Q_TABLES).mkdir(parents=True, exist_ok=True)

env = gym.make("MountainCar-v0")
env.reset()

LEARNING_RATE = 0.1
DISCOUNT = 0.95  # Measure of how important are future actions vs current actions, or future rewards vs current rewards
EPISODES = 50000
SHOW_EVERY = 3000
STATS_EVERY = 100

DISCRETE_OS_SIZE = [40] * len(env.observation_space.high)  # Discrete Observation Space
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

# Exploration settings
# "epsilon" is not a constant, going to be decayed
epsilon = 1  # between 0 and 1: Measure of how much exploration/randomness we want, you may wanna decay it over time
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2  # // means always divide to an integer
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

# A table of 20 by 20 by 3; where 3 is the size of action_space of environment
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

# For stats
ep_rewards = []  # List that contains rewards for each episode
# For graphing, dictionary that tracks, 1: episode numbers, 2: trailing average for any given window e.g. for every
# 500 episodes, 3: tracking worst model, 4: best model so far
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}


def get_discrete_state(state):
    _discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    # we use this tuple to look up the 3 Q values for the available actions in the q-table
    return tuple(_discrete_state.astype(int))


for episode in range(1, EPISODES + 1):
    episode_reward = 0

    discrete_state = get_discrete_state(env.reset())
    done = False

    if episode % SHOW_EVERY == 0:
        render = True
        print(episode)
    else:
        render = False

    while not done:
        if np.random.random() > epsilon:
            # Get action from Q table
            # action==0: push car left
            # action==1: do nothing
            # action==2: push car right
            action = np.argmax(q_table[discrete_state])
        else:
            # Get random action, explore more solutions
            action = np.random.randint(0, env.action_space.n)

        new_state, reward, done, _ = env.step(action)

        episode_reward += reward

        new_discrete_state = get_discrete_state(new_state)

        if render:
            env.render()

        # If simulation did not end yet after last step - update Q table
        if not done:

            # Maximum possible Q value in next step (for new state)
            max_future_q = np.max(q_table[new_discrete_state])

            # Current Q value (for current state and performed action)
            current_q = q_table[discrete_state + (action, )]

            # And here's our equation for a new Q value for current state and action
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

            # Update Q table with new Q value
            q_table[discrete_state + (action, )] = new_q

        # Simulation ended (for any reason) - if goal position is achieved - update Q value with reward directly
        elif new_state[0] >= env.goal_position:  # new_state[0]: position, new_state[1]: velocity
            # print(f"We made it on episode {episode}")
            # q_table[discrete_state + (action,)] = reward
            q_table[discrete_state + (action, )] = 0

        discrete_state = new_discrete_state

    # Decaying is being done every episode if possible number is within decaying range
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

    ep_rewards.append(episode_reward)

    if not episode % STATS_EVERY:
        average_reward = sum(ep_rewards[-STATS_EVERY:]) / STATS_EVERY
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['max'].append(max(ep_rewards[-STATS_EVERY:]))
        aggr_ep_rewards['min'].append(min(ep_rewards[-STATS_EVERY:]))

        print(
            f"Episode: {episode:>5d} average reward: {average_reward:>4.1f}, "
            f"current epsilon: {epsilon:>1.2f}"
        )

    if episode % 10 == 0:
        np.save(f"{DIR_Q_TABLES}/{episode}-qtable.npy", q_table)

    env.close()

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label='avg rewards')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label='max rewards')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label='min rewards')
plt.legend(loc=4)
plt.grid(True)
plt.show()
