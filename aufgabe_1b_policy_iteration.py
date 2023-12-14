import gymnasium as gym
import numpy as np

# env.observation_space.n is a number of states in the environment.
# env.action_space.n is a number of actions in the environment.

# Environment init
#render_mode = "human"
render_mode = "ansi"
env = gym.make(
    "FrozenLake-v1", render_mode=render_mode, desc=None, map_name="4x4", is_slippery=False
)
# env = gym.make("Taxi-v3", render_mode=render_mode)


# Parameter init
num_episodes = 10000  # Total number of episodes to play during training
max_steps_per_episode = 100
discount_factor = 0.99


def compute_value_function(policy):
    value_table = np.zeros(env.observation_space.n)
    threshold = 1e-10
    while True:
        updated_value_table = np.copy(value_table)
        for state in range(env.observation_space.n):
            action = policy[state]
            value_table[state] = sum(
                [
                    trans_prob * (reward_prob + discount_factor * updated_value_table[next_state])
                    for trans_prob, next_state, reward_prob, _ in env.P[state][action]
                ]
            )
        if np.sum((np.fabs(updated_value_table - value_table))) <= threshold:
            break
    return value_table


def extract_policy(value_table):
    policy = np.zeros(env.observation_space.n)
    for state in range(env.observation_space.n):
        Q_table = np.zeros(env.action_space.n)
        for action in range(env.action_space.n):
            for next_sr in env.P[state][action]:
                trans_prob, next_state, reward_prob, _ = next_sr
                Q_table[action] += trans_prob * (
                    reward_prob + discount_factor * value_table[next_state]
                )
        policy[state] = np.argmax(Q_table)

    return policy


def policy_iteration():
    state = env.reset()[0]  # Reset back to starting state
    random_policy = np.zeros(env.observation_space.n)
    for i in range(num_episodes):
        new_value_function = compute_value_function(random_policy)
        new_policy = extract_policy(new_value_function)
        if np.all(random_policy == new_policy):
            print("Policy-Iteration converged at step %d." % (i + 1))
            break
        random_policy = new_policy
        env.render()
    return new_policy


opt_Policy = policy_iteration()
print("Converged")
print("Final Policy: ")
print(opt_Policy)
