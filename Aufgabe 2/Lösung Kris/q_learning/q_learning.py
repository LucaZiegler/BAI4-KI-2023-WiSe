import random
import numpy as np


def q_learning(_env, _qtable, _num_iterations, _epsilon, _discount_rate, _learning_rate, _decay_rate):
    counter_explore = 0
    counter_exploit = 0

    # training
    for _ in range(_num_iterations):
        # reset the environment
        state, _ = _env.reset()  # dont use info
        terminated = False

        while not terminated:

            # exploration vs exploitation
            if random.uniform(0, 1) < _epsilon:
                # explore
                action = _env.action_space.sample()
                counter_explore += 1
            else:
                # exploit
                action = np.argmax(_qtable[state, :])
                counter_exploit += 1

            # take action and observe reward
            new_state, reward, terminated, truncated, _ = _env.step(action)

            if truncated:
                print("Truncated: reset state")
                state, _ = _env.reset()  # dont use info
                break

            # Q-learning algorithm
            _qtable[state, action] = (
                    _qtable[state, action] +  # (1-alpha) * Q(s,a) +
                    _learning_rate *  # alpha * [ R(s,a,s’) + gamma * max’Q(s’,a’) ]
                    (
                            reward +
                            _discount_rate * np.max(_qtable[new_state, :]) -
                            _qtable[state, action]
                    )
            )

            # Update to our new state
            state = new_state

        # Decrease epsilon
        _epsilon = max(_epsilon - _decay_rate, 0)

    print(f"Training completed over {_num_iterations} episodes")
    print(f"\r\nExploited: {counter_exploit}; Explored: {counter_explore}")