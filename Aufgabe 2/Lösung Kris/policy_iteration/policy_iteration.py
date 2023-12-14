import numpy as np


def evaluate_policy(_env, policy, _discount_rate, _epsilon):
    state_size = _env.observation_space.n

    # initialize value function
    V = np.zeros(state_size)

    while True:
        delta = 0

        for state in range(state_size):
            v = V[state]
            action = policy[state]
            # update value function using the current policy
            values = 0
            for p, next_state, r, _ in _env.P[state][action]:
                values += p * (r + _discount_rate * V[next_state])
            V[state] = values
            delta = max(delta, abs(v - V[state]))

        if delta < _epsilon:
            break

    return V


def improve_policy(_env, V, _discount_rate):
    state_size = _env.observation_space.n
    action_size = _env.action_space.n

    new_policy = np.zeros(state_size, dtype=int)

    for state in range(state_size):
        action_values = []

        # Calculate action values for each action in the current state
        for action in range(action_size):
            # Calculate the action value for the current action
            action_value = 0
            for p, next_state, r, _ in _env.P[state][action]:
                action_value += p * (r + _discount_rate * V[next_state])

            # Append the action value to the list
            action_values.append(action_value)

        # Select the action with the maximum value for the current state
        new_policy[state] = np.argmax(action_values)

    return new_policy


def policy_iteration(_env, _num_iterations, _discount_rate, _epsilon):
    state_size = _env.observation_space.n
    action_size = _env.action_space.n

    # initialize a random policy
    policy = np.random.choice(action_size, state_size)

    for _ in range(_num_iterations):
        # Policy Evaluation - return new value function
        V = evaluate_policy(_env, policy, _discount_rate, _epsilon)

        # Policy Improvement
        new_policy = improve_policy(_env, V, _discount_rate)

        # Check for convergence
        if np.array_equal(policy, new_policy):
            break

        policy = new_policy

    return V, policy