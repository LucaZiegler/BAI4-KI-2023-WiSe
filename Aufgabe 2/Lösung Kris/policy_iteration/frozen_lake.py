import numpy as np
import gym

from src.policy_iteration.policy_iteration import policy_iteration


def createOptimalPolicy(policy):
    optimalPolicy = []

    for field in policy:
        optimalPolicy.append(action_to_str(field))

    array_1d = np.array(optimalPolicy)
    array_2d = array_1d.reshape((4, 4))

    return array_2d


def action_to_str(action):
    return ["left", "down", "right", "up"][action]


def mark_zero_values(_optimal_values, _optimal_policy):
    marked_policy = np.copy(_optimal_policy)

    for i in range(_optimal_values.shape[0]):
        for j in range(_optimal_values.shape[1]):
            if _optimal_values[i, j] == 0:
                marked_policy[i, j] = 'X'

    return marked_policy


if __name__ == "__main__":
    env = gym.make('FrozenLake-v1', is_slippery=True)

    # policy iteration hyperparameters
    num_iterations = 10000
    discount_rate = 0.9
    epsilon = 1e-8

    # perform policy iteration
    optimal_values, optimal_policy = policy_iteration(env, num_iterations, discount_rate, epsilon)

    optimal_value_2d = optimal_values.reshape(4, 4)
    optimal_policy_2d = createOptimalPolicy(optimal_policy).reshape(4,4)

    print("Optimal Values:")
    print(optimal_value_2d)

    print("Optimal Policy:")
    print(mark_zero_values(optimal_value_2d, optimal_policy_2d))


