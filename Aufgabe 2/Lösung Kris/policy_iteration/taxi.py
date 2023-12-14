import sys
import numpy as np
import gym

from src.policy_iteration.policy_iteration import policy_iteration
from src.utils.Strategy import Strategy
from src.utils.utils_taxi import watchTrainedAgent, calculatePathByOptimalPolicy

if __name__ == "__main__":
    env = gym.make('Taxi-v3', render_mode="ansi")

    # policy iteration hyperparameters
    num_iterations = 10000
    discount_rate = 0.9
    epsilon = 1e-8

    # perform policy iteration
    optimal_values, optimal_policy = policy_iteration(env, num_iterations, discount_rate, epsilon)

    print("Optimal Values:")
    print(optimal_values)

    print("Optimal Policy:")
    print(optimal_policy)

    watchTrainedAgent(num_iterations, optimal_policy, env, Strategy.POLICY_ITERATION)
    calculatePathByOptimalPolicy(optimal_policy, env, Strategy.POLICY_ITERATION)
