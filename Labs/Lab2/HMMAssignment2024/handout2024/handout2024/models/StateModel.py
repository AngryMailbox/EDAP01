# The state model describes the dimensions of the grid and provides methods to transfrom a pose (x, y, h)
# or a position (x, y) into a state i or a sensor reading r and/or vice versa. x runs over rows, y over
# columns. "State" means the index coding for a pose, "reading" means the index coding for a position.
#
# State 0 = Pose < x=0, y=0, h=0> is the upper left corner of the grid direction "SOUTH", rows run then down,
# columns to the right, headings go COUNTER-clockwise around the cell. Headings run first, i.e.
# state 1 = Pose < x=0, y=0, h=1> (upper left corner, facing east)
# state 2 = Pose < x=0, y=0, h=2> (upper left corner, facing north)
#
# Seen from above (from pose <0, 0, 0>), this corresponds to a robot coordinate frame, where the x
# axis is "forward", and y means "left".
# Seen from the side (from pose <0, 0, 1>), you get the "regular" coordinate system with the x-axis running
# to the right and the y-axis running "up".
#
# The encoding means that there will be numberOfRows*numberOfCols*4 states (poses) and numberOfRows*numberOfCols+1
# possible sensor readings (numberOfRows*numberOfCols positions + "none")

import numpy as np
import matplotlib.pyplot as plt
import random


class StateModel:

    def __init__(self, rows, cols):
        """Initializes the state model with the given grid dimensions."""
        self.__rows = rows
        self.__cols = cols
        self.__head = 4
        self.__num_states = rows * cols * 4
        self.__num_readings = rows * cols + 1

    def state_to_pose(self, s: int) -> (int, int, int):
        """Returns the pose (x, y, h) corresponding to the state index s"""
        x = s // (self.__cols * self.__head)
        y = (s - x * self.__cols * self.__head) // self.__head
        h = s % self.__head

        return x, y, h

    def pose_to_state(self, x: int, y: int, h: int) -> int:
        """Returns the state index corresponding to the pose (x, y, h)"""
        return x * self.__cols * self.__head + y * self.__head + h

    def state_to_position(self, s: int) -> (int, int):
        """Returns the position (x, y) corresponding to the state index s"""
        x = s // (self.__cols * self.__head)
        y = (s - x * self.__cols * self.__head) // self.__head

        return x, y

    def reading_to_position(self, r: int) -> (int, int):
        """Returns the position (x, y) corresponding to the sensor reading r"""
        x = r // self.__cols
        y = r % self.__cols

        return x, y

    def position_to_reading(self, x: int, y: int) -> int:
        """Returns the sensor reading corresponding to the position (x, y)"""
        return x * self.__cols + y

    def state_to_reading(self, s: int) -> int:
        """Returns the sensor reading corresponding to the state index s"""
        return s // self.__head

    # Note that a reading contains less information than a state, i.e., this will always give the state
    # corresponding to heading=0 (SOUTH) in the cell that corresponds to "reading r"
    def reading_to_ref_state(self, r: int) -> int:
        """Returns the state index corresponding to the sensor reading r"""
        return r * self.__head

    def get_grid_dimensions(self) -> (int, int, int):
        """Returns the grid dimensions (rows, cols, head) of the state model."""
        return self.__rows, self.__cols, self.__head

    def get_num_of_states(self) -> int:
        """Returns the number of states in the state model."""
        return self.__num_states

    def get_num_of_readings(self) -> int:
        """Returns the number of readings in the state model."""
        return self.__num_readings
