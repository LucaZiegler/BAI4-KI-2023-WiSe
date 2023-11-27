import numpy as np
import gym
import random

from q_table_navigator import QTableNavigator

# env.nS is a number of states in the environment.
# env.nA is a number of actions in the environment.

# Environment init
# env = gym.make("FrozenLake-v1", render_mode="human")
env = gym.make("FrozenLake-v1", render_mode="ansi", is_slippery=False)
# env = gym.make("Taxi-v3", render_mode="ansi")


# Parameter init
action_space_size = env.action_space.n
state_space_size = env.observation_space.n

q_table = np.zeros((state_space_size, action_space_size))

num_episodes = 1000  # Total number of episodes to play during training
max_steps_per_episode = 1000  #

learning_rate = 0.8  # Alpha
discount_rate = 0.9  # Gamma

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.001  # 0.01 / 0.001

reward_all_episodes = []


# Q Learning algorithm / training
for episode in range(num_episodes):
    state = env.reset()[0]  # Reset back to starting state

    done = False  # Later for episode is over
    rewards_current_episode = 0  # No rewards in beginning

    for step in range(max_steps_per_episode):
        # Exploration-exploitation trade-off
        exploration_rate_threshold = random.uniform(
            0, 1
        )  # Random number between 0 and 1
        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(q_table[state, :])
        else:
            action = env.action_space.sample()  # Gives random valid action

        new_state, reward, done, truncated, info = env.step(action)

        # Update Q-Table for Q(s,a)
        q_table[state, action] = q_table[state, action] * (
            1 - learning_rate
        ) + learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))

        state = new_state
        rewards_current_episode += reward

        if done == True:
            break
        env.render()
    # Exploration rate decay
    exploration_rate = min_exploration_rate + (
        max_exploration_rate - min_exploration_rate
    ) * np.exp(-exploration_decay_rate * episode)
    reward_all_episodes.append(rewards_current_episode)

# Statistics
rewards_per_thousand_episodes = np.split(
    np.array(reward_all_episodes), num_episodes / 1000
)
count = 1000
print("****AVERAGE reward per thousand episodes****\n")
for r in rewards_per_thousand_episodes:
    print(count, ": ", str(sum(r / 1000)))
    count += 1000

# Print Updated Q-Table
print("\n\n*** Q-Table ***\n")
print("LEFT DOWN RIGHT UP")
print(q_table)


table_nav = QTableNavigator(q_table, 4)
path = table_nav.get_path(x=0, y=0, x_end=3, y_end=3, path_array=[])

print("\n\nBEST PATH")
print(path)
