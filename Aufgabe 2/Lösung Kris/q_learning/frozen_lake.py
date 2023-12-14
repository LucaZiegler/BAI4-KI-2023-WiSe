import sys
import gym
import numpy as np
from gym import *

from src.q_learning.q_learning import q_learning


def watchTrainedAgent(_num_iterations, _qtable):
    # watch trained agent
    _env: gym.Env = gym.make('FrozenLake-v1', is_slippery=True, render_mode="human")
    state, _ = _env.reset()
    rewards = 0

    for s in range(_num_iterations):

        print(f"TRAINED AGENT")
        print("Step {}".format(s + 1))

        action = np.argmax(_qtable[state, :])
        new_state, reward, terminated, truncated, _ = _env.step(action)
        rewards += reward
        print(f"score: {rewards}")
        state = new_state

        if terminated:
            if reward < 1.0:
                print("You're dead.")
                break
            else:
                print("You won - gg.")
                break

    np.set_printoptions(threshold=sys.maxsize)


def createOptimalPolicy(qtable):
    optimalPolicy = []

    for field in qtable:
        max_index = np.argmax(field)
        if np.max(field) == 0.0:
            optimalPolicy.append("X")
        else:
            optimalPolicy.append(action_to_str(max_index))

    array_1d = np.array(optimalPolicy)
    array_2d = array_1d.reshape((4, 4))

    return array_2d


def action_to_str(action):
    return ["left", "down", "right", "up"][action]


if __name__ == "__main__":
    env: gym.Env = gym.make('FrozenLake-v1', is_slippery=True)

    # hyperparameters
    learning_rate = 0.92  # alpha
    discount_rate = 0.95  # gamma, discount factor to give more or less importance to the next reward
    epsilon = 0.9  # explore vs exploit
    decay_rate = 0.005  # Fixed amount to decrease epsilon
    num_iterations = 5000

    state_size: Space = env.observation_space.n
    action_size: Space = env.action_space.n
    qtable = np.zeros((state_size, action_size))

    q_learning(env, qtable, num_iterations, discount_rate, epsilon, learning_rate, decay_rate)
    # watchTrainedAgent(num_iterations, qtable)

    optimal_policy = createOptimalPolicy(qtable)

    print("\r\nOptimal Policy:")
    print(optimal_policy.reshape((4, 4)))
