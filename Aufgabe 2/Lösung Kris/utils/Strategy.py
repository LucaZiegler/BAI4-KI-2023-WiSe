from enum import Enum


class Strategy(Enum):
    POLICY_ITERATION = 1
    VALUE_ITERATION = 2
    Q_LEARNING = 3
