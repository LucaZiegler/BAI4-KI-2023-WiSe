from numpy import ndarray
import numpy


class QTableNavigator:
    def __init__(self, table: ndarray, col_count: int):
        self.table = table
        self.col_count = col_count

    def get_path(self, x: int, y: int, x_end: int, y_end: int, path_array: list):
        path_array.append([x, y])

        if x == x_end & y == y_end:
            return path_array  # Goal reached

        list_idx = (y * self.col_count) + x

        self.cur_state = self.table[list_idx]
        idx = self.cur_state.argmax()
        if idx == 0:
            return self.get_path((x - 1), y, x_end, y_end, path_array)
        elif idx == 1:
            return self.get_path(x, (y + 1), x_end, y_end, path_array)
        elif idx == 2:
            return self.get_path((x + 1), y, x_end, y_end, path_array)
        elif idx == 3:
            return self.get_path(x, (y - 1), x_end, y_end, path_array)
