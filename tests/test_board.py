import numpy as np

from sgf_solver.board import GoBoard, BOARD_SHAPE


board_get_loc_1 = np.zeros(BOARD_SHAPE)
board_get_loc_1[0, 0] = 1
board_get_loc_1[-1, -1] = -1


def test_get_loc():
    pass