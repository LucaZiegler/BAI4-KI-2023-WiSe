import gym
import numpy as np
from gym import *

from src.q_learning.q_learning import q_learning
from src.utils.Strategy import Strategy
from src.utils.utils_taxi import watchTrainedAgent, calculatePathByOptimalPolicy


if __name__ == "__main__":
    env: gym.Env = gym.make('Taxi-v3', render_mode="ansi")
    # hyperparameters
    learning_rate = 0.92  # alpha
    discount_rate = 0.95  # gamma, discount factor to give more or less importance to the next reward
    epsilon = 0.9  # explore vs exploit
    decay_rate = 0.005  # Fixed amount to decrease epsilon
    num_iterations = 10000

    state_size: Space = env.observation_space.n
    action_size: Space = env.action_space.n
    qtable = np.zeros((state_size, action_size))

    q_learning(env, qtable, num_iterations, discount_rate, epsilon, learning_rate, decay_rate)
    watchTrainedAgent(num_iterations, qtable, env, Strategy.Q_LEARNING)
    calculatePathByOptimalPolicy(qtable, env, Strategy.Q_LEARNING)


