import sys
import numpy as np

from src.utils.Strategy import Strategy

states_to_calculate: dict = {
    "red -> green": [0, 0, 0, 1],
    "red -> yellow": [0, 0, 0, 2],
    "red -> blue": [0, 0, 0, 3],
    "green -> red": [4, 0, 1, 0],
    "green -> yellow": [4, 0, 1, 2],
    "green -> blue": [4, 0, 1, 3],
    "yellow -> red": [0, 4, 2, 0],
    "yellow -> green": [0, 4, 2, 1],
    "yellow -> blue": [0, 4, 2, 3],
    "blue -> red": [3, 4, 3, 0],
    "blue -> green": [3, 4, 3, 1],
    "blue -> yellow": [3, 4, 3, 2],
}


def watchTrainedAgent(_num_iterations, _optimal_policy, _env, strategy: Strategy):
    # watch trained agent
    state, _ = _env.reset()
    rewards = 0

    for s in range(_num_iterations):

        print(f"TRAINED AGENT")
        print("Step {}".format(s + 1))

        if strategy == Strategy.POLICY_ITERATION:
            action = _optimal_policy[state]
        else:
            action = np.argmax(_optimal_policy[state, :])

        new_state, reward, terminated, truncated, _ = _env.step(action)
        rewards += reward
        print(f"score: {rewards}")
        state = new_state
        _env.s = state
        print(f"state: {_env.s}")
        p = _env.render()
        print(p)

        if terminated:
            if reward < 1.0:
                print("You're dead.")
                break
            else:
                print("You won - gg.")
                break

    np.set_printoptions(threshold=sys.maxsize)


def calculatePathByOptimalPolicyHelper(taxi_row, taxi_column, passenger_location, destination, optimal_policy, env,
                                       strategy: Strategy):
    passenger_matrix = np.array([[0, 0], [4, 0], [0, 4], [3, 4]])
    pathByOptimalPolicy = []
    env.reset()

    while True:
        state = env.encode(taxi_row, taxi_column, passenger_location, destination)
        pathByOptimalPolicy.append(state)

        if strategy == Strategy.POLICY_ITERATION:
            action = optimal_policy[state]
        else:
            action = np.argmax(optimal_policy[state, :])

        match action:
            case 0:
                taxi_row += 1
            case 1:
                taxi_row -= 1
            case 2:
                taxi_column += 1
            case 3:
                taxi_column -= 1
            case 4:
                if passenger_location == 4:
                    raise Exception("Taxi trying to pick up passenger who is already inside the car.")

                if np.array_equal(passenger_matrix[passenger_location], [taxi_row, taxi_column]):
                    passenger_location = 4
                else:
                    raise Exception("Taxi trying to pick up passenger who is not on the designated pick up field.")
            case 5:
                if passenger_location != 4:
                    raise Exception("Taxi trying to drop off passenger who is not inside the car.")

                if not np.array_equal([taxi_row, taxi_column], passenger_matrix[destination]):
                    state = env.encode(taxi_row, taxi_column, passenger_location, destination)
                    pathByOptimalPolicy.append(state)

                    break
                else:
                    raise Exception("Taxi trying to drop off passenger at the wrong destination.")

    return pathByOptimalPolicy


def calculatePathByOptimalPolicy(_qtable, _env, strategy: Strategy):
    for key in states_to_calculate:
        taxi_row, taxi_column, passenger_location, destination = states_to_calculate[key]
        path = (
            calculatePathByOptimalPolicyHelper(
                taxi_row, taxi_column, passenger_location, destination, _qtable, _env, strategy)
        )
        print(f"Optimal path for {key}: {path}")