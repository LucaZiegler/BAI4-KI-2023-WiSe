import gymnasium as gym
import numpy as np

# env.observation_space.n is a number of states in the environment.
# env.action_space.n is a number of actions in the environment.

# Environment init
env_type = 0

# render_mode = "human"
render_mode = "ansi"
if env_type == 0:
    env = gym.make(
        "FrozenLake-v1", render_mode=render_mode, map_name="4x4", is_slippery=True
    )
elif env_type == 1:
    env = gym.make("Taxi-v3", render_mode=render_mode)

# Parameter init
num_episodes = 100  # Total number of episodes to play during training
max_steps_per_episode = 100
discount_factor = 0.95  # from 0.95 to 0.99 for taxi game


def one_step_lookahead(state, V):
    """
    Helper function to  calculate state-value function

    Arguments:
        state: state to consider
        V: Estimated Value for each state. Vector of length nS

    Return:
        action_values: Expected value of each action in a state. Vector of length nA
    """

    # initialize vector of action values
    action_values = np.zeros(env.action_space.n)

    # loop over the actions we can take in an enviorment
    for action in range(env.action_space.n):
        # loop over the P_sa distribution.
        for probablity, next_state, reward, info in env.P[state][action]:
            # if we are in state s and take action a. then sum over all the possible states we can land into.
            action_values[action] += probablity * (
                reward + (discount_factor * V[next_state])
            )

    return action_values


def update_policy(policy, V):
    """
    Helper function to update a given policy based on given value function.

    Arguments:
        policy: policy to update.
        V: Estimated Value for each state. Vector of length nS.
    Return:
        policy: Updated policy based on the given state-Value function 'V'.
    """

    for state in range(env.observation_space.n):
        # for a given state compute state-action value.
        action_values = one_step_lookahead(state, V)

        # choose the action which maximizez the state-action value.
        policy[state] = np.argmax(action_values)
    return policy


def value_iteration():
    """
    Algorithm to solve MPD.

    Return:
        V: Optimal state-Value function. Vector of lenth nS.
        optimal_policy: Optimal policy. Vector of length nS.

    """
    # intialize value fucntion
    V = np.zeros(env.observation_space.n, dtype=np.float64)
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    # iterate over max_iterations
    for i in range(num_episodes):
        #  keep track of change with previous value function
        prev_v = np.copy(V)

        # loop over all states
        for state in range(num_states):
            # Asynchronously update the state-action value
            # action_values = one_step_lookahead(env, state, V, discount_factor)

            # Synchronously update the state-action value
            action_values = one_step_lookahead(state, prev_v)

            # select best action to perform based on highest state-action value
            best_action_value = np.max(action_values)

            # update the current state-value fucntion
            V[state] = best_action_value

        # if policy not changed over 10 iterations it converged.
        if i % 10 == 0:
            # if values of 'V' not changing after one iteration
            if np.all(np.isclose(V, prev_v)):
                print("Value converged at iteration %d" % (i + 1))
                break

    # intialize optimal policy
    optimal_policy = np.zeros(env.observation_space.n, dtype="int8")

    # update the optimal polciy according to optimal value function 'V'
    optimal_policy = update_policy(optimal_policy, V)

    return V, optimal_policy


env.reset()
env.render()

opt_V, opt_Policy = value_iteration()
print("Converged")
print("Optimal Value function: ")

if env_type == 0:
    print(opt_V.reshape((env.action_space.n, env.action_space.n)))

print("Final Policy: ")
print(opt_Policy.reshape(4,4))
# print(" ".join([action_mapping[int(action)] for action in opt_Policy]))
