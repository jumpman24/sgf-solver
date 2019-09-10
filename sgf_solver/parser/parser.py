import numpy as np

from sgf_solver.board import CoordType, GoBoard
from sgf_solver.constants import Location, BOARD_SHAPE
from sgf_solver.parser.sgflib import Node, GameTree, SGFParser


def parse_sgf_file(filepath: str) -> GameTree:
    with open(filepath) as file:
        sgf_data = file.read()

    return SGFParser(sgf_data).parse()[0]


def get_coord(sgf_coord: str) -> CoordType:
    if len(sgf_coord) != 2:
        raise Exception("Wrong coordinates")

    return ord(sgf_coord[1]) - 97, ord(sgf_coord[0]) - 97


def get_board_data(node: Node):
    black = node.get('AB').data
    white = node.get('AW').data

    board = np.zeros(BOARD_SHAPE, dtype=int)

    for stone in black:
        coord = get_coord(stone)
        board[coord] = Location.BLACK

    for stone in white:
        coord = get_coord(stone)
        board[coord] = Location.WHITE

    return board


def get_trees(game_tree: GameTree):
    if not game_tree.variations:
        return [game_tree.data, ]

    trees = []
    common_data = game_tree.data
    for variation in game_tree.variations:
        nested_trees = get_trees(variation)

        for nt in nested_trees:
            trees.append(common_data + nt)

    return trees


def parse_tree(board, tree: list):
    comment = tree[-1].get('C')

    if comment.value.startswith('Correct'):
        initial_value = 1
        print("Correct tree found")
    elif comment.value.startswith('Wrong'):
        initial_value = 0
        print("Wrong tree found")
    else:
        return None

    brd = GoBoard(board)


if __name__ == '__main__':
    filepath = '/home/alex/PycharmProjects/sgf-solver/tests/sgf_001.sgf'

    tree = parse_sgf_file(filepath)

    board = get_board_data(tree[0])
    trees = get_trees(tree)
    print(board)

    for tree in trees:
        parse_tree(tree)
