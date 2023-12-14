import numpy as np
import gym

from src.utils.Strategy import Strategy
from src.utils.utils_taxi import watchTrainedAgent, calculatePathByOptimalPolicy


def value_iteration(_env, _num_iterations, _epsilon, _discount_rate):
    state_size = _env.observation_space.n
    action_size = _env.action_space.n

    # initialize value function
    _V = np.zeros(state_size)

    for _ in range(_num_iterations):
        delta = 0

        for state in range(state_size):
            current_state = _V[state]
            A = np.zeros(action_size)

            for action in range(action_size):
                for probability, next_state, reward, _ in _env.P[state][action]:
                    A[action] += probability * (reward + _discount_rate * _V[next_state])

            _V[state] = max(A)

            delta = max(delta, abs(current_state - _V[state]))

        # delta converges to epsilon -> stop
        if delta < _epsilon:
            break

    _optimal_policy = np.zeros([state_size, action_size])

    for state in range(state_size):  # for all states, create deterministic policy

        A = np.zeros(action_size)
        for action in range(action_size):
            for probability, next_state, reward, _ in _env.P[state][action]:
                A[action] += probability * (reward + _discount_rate * _V[next_state])

        best_action = np.argmax(A)
        _optimal_policy[state][best_action] = 1

    return _V, _optimal_policy


if __name__ == "__main__":
    env = gym.make('Taxi-v3', render_mode='ansi')

    # value iteration hyperparameters
    num_iterations = 10000
    epsilon = 0.00000001  # converges to epsilon
    discount_rate = 0.8

    # perform value iteration
    V, optimal_policy = value_iteration(env, num_iterations, epsilon, discount_rate)

    print("Optimal Values:")
    print(V)

    print("Optimal Policy:")
    print(optimal_policy)

    watchTrainedAgent(num_iterations, optimal_policy, env, Strategy.VALUE_ITERATION)
    calculatePathByOptimalPolicy(optimal_policy, env, Strategy.VALUE_ITERATION)