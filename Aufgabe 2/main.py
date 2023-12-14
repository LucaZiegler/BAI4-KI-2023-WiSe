import time
import gymnasium as gym
import numpy as np



def value_iterationOriginal(S, A, P, R):
    """
    :param list S: set of states
    :param list A: set of actions
    :param function P: transition function
    :param function R: reward function
    """
    V = {s: 0 for s in S}
    optimal_policy = {s: 0 for s in S}
    while True:
        oldV = V.copy()

        for s in S:
            Q = {}
            for a in A:
                Q[a] = R(s, a) + sum(P(s_next, s, a) * oldV[s_next] for s_next in S)
            V[s] = max(Q.values())
            optimal_policy[s] = max(Q, key=Q.get)
        if all(oldV[s] == V[s] for s in S):
            break
    return V, optimal_policy

def value_iteration(env):
    """
    :param list S: set of states
    :param list A: set of actions
    :param function P: transition function
    :param function R: reward function
    """

def one_step_lookahead(environment, state, V, discount_factor):
    action_values = np.zeros(environment.nA)
    for action in range(environment.nA):
        for probability, next_state, reward, terminated in environment.P[state][action]:
            action_values[action] += probability * (reward + discount_factor * V[next_state])
    return action_values

def value_iteration(environment, discount_factor=1.0, theta=1e-9, max_iterations=1e9):
    # Initialize state-value function with zeros for each environment state
    V = np.zeros(environment.nS)
    for i in range(int(max_iterations)):
        # Early stopping condition
        delta = 0
        # Update each state
        for state in range(environment.nS):
            # Do a one-step lookahead to calculate state-action values
            action_value = one_step_lookahead(environment, state, V, discount_factor)
            # Select best action to perform based on the highest state-action value
            best_action_value = max(action_value)
            # Calculate change in value
            delta = max(delta, np.abs(V[state] - best_action_value))
            # Update the value function for current state
            V[state] = best_action_value
            # Check if we can stop
        if delta < theta:
            print(f'Value-iteration converged at iteration #{i}.')
            break

    # Create a deterministic policy using the optimal value function
    policy = np.zeros([environment.nS, environment.nA])
    for state in range(environment.nS):
        # One step lookahead to find the best action for this state
        action_value = one_step_lookahead(environment, state, V, discount_factor)
        # Select best action based on the highest state-action value
        best_action = action_value.tolist().index(max(action_value))
        # Update the policy to perform a better action at a current state
        policy[state, best_action] = 1.0
    return policy, V


env = gym.make("FrozenLake-v1",render_mode="human", desc=None, map_name="4x4", is_slippery=True)

env.reset()
value_iteration(env.env)


# V, optimal_policy = value_iteration(env)
