import numpy as np

from .sgflib import Node


def get_coord(s):
    if len(s) != 2:
        raise Exception("Wrong coordinates")
    x = ord(s[0]) - 97
    y = ord(s[1]) - 97
    return y, x


def get_board_data(node: Node):
    if 'SZ' in node:
        sz = int(node['SZ'].value)
    else:
        sz = 19

    black = node.get('AB').data
    white = node.get('AW').data

    board = np.zeros((19, 19))

    for stone in black:
        coord = get_coord(stone)
        board[coord] = 1

    for stone in white:
        coord = get_coord(stone)
        board[coord] = -1

    return board, sz
