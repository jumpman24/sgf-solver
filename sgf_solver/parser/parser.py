import os

import h5py
import numpy as np

from sgf_solver.board import CoordType, GoBoard
from sgf_solver.constants import Location, BOARD_SHAPE, PROBLEM_DATASET, PROBLEM_PATH
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


def order_trees(trees: list):
    correct, wrong = [], []

    for tree in trees:
        comment = tree[-1].get('C')
        if comment.value.startswith('Correct'):
            correct.append(tree)
        elif comment.value.startswith('Wrong'):
            wrong.append(tree)

    return correct + wrong


def parse_tree(tree: list):
    comment = tree[-1].get('C')

    if comment.value.startswith('Correct'):
        value = 1
    elif comment.value.startswith('Wrong'):
        value = 0
    else:
        return [], [], []

    board = GoBoard(get_board_data(tree[0]))

    problems = []
    values = []
    answers = []
    color = 'B'
    for node in tree[1:]:
        coord = get_coord(node.get(color).value)

        if not coord:
            raise Exception("No move in move node")

        # find the next move
        answer = np.zeros(BOARD_SHAPE, dtype=int)
        answer[coord] = 1

        # store problem, value and answers
        problems.append(board.board if color == 'B' else board.board * -1)
        answers.append(answer)
        values.append(value)

        # make a move
        board.move(coord)

        color = 'B' if color == 'W' else 'W'
        value = 1 - value

    return problems, values, answers


def flip_transpose(problems, values, answers):
    new_boards, new_values, new_answers = [], [], []

    for board, value, answer in zip(problems, values, answers):
        new_boards.extend([
            np.array(board),
            np.flip(board, axis=(0,)),
            np.flip(board, axis=(1,)),
            np.flip(board, axis=(0, 1)),
            np.transpose(board),
            np.transpose(np.flip(board, axis=(0,))),
            np.transpose(np.flip(board, axis=(1,))),
            np.transpose(np.flip(board, axis=(0, 1)))])
        new_values.extend([value, value, value, value, value, value, value, value])
        new_answers.extend([
            np.array(answer),
            np.flip(answer, axis=(0,)),
            np.flip(answer, axis=(1,)),
            np.flip(answer, axis=(0, 1)),
            np.transpose(answer),
            np.transpose(np.flip(answer, axis=(0,))),
            np.transpose(np.flip(answer, axis=(1,))),
            np.transpose(np.flip(answer, axis=(0, 1)))])

    return new_boards, new_values, new_answers


def sgf_to_problems(sgf: GameTree):
    tree_list = order_trees(get_trees(sgf))

    sgf_problems = []
    sgf_values = []
    sgf_answers = []
    correct_hashes = set()

    for tree in tree_list:
        hashes = []
        problems, values, answers = [], [], []

        for problem, value, answer in zip(*parse_tree(tree)):
            # merge another answer to existing problems
            problem_hash = hash(str(problem))
            if problem_hash in hashes:
                answers[hashes.index(problem_hash)] += answer
                continue

            if value == 1:
                correct_hashes.add(problem_hash)
            elif problem_hash in correct_hashes:
                continue

            hashes.append(problem_hash)
            problems.append(problem)
            values.append(value)
            answers.append(answer)

        problems, values, answers = flip_transpose(problems, values, answers)
        sgf_problems.extend(problems)
        sgf_values.extend(values)
        sgf_answers.extend(answers)

    return sgf_problems, sgf_values, sgf_answers


if __name__ == '__main__':
    all_sgf_files = []
    for directory, _, files in os.walk(PROBLEM_PATH):
        for filename in files:
            if os.path.splitext(filename)[1] == '.sgf':
                all_sgf_files.append(os.path.join(directory, filename))

    all_problems, all_values, all_answers = [], [], []

    good_count, bad_count = 0, 0
    for filepath in sorted(all_sgf_files):
        tree = parse_sgf_file(filepath)
        problems, values, answers = sgf_to_problems(tree)

        all_problems.extend(problems)
        all_values.extend(values)
        all_answers.extend(answers)

        good_count += sum(values)
        bad_count += len(problems) - sum(values)
        print(f"\rProblems created: {good_count + bad_count:-6d}, "
              f"good: {good_count:-4d}, bad: {bad_count:-4d}", end='')
    print()

    with h5py.File(PROBLEM_DATASET, 'w') as dataset:
        dataset.create_dataset('problems', data=all_problems)
        dataset.create_dataset('values', data=all_values)
        dataset.create_dataset('answers', data=all_answers)
        print(dataset['problems'].shape)
        print(dataset['values'].shape)
        print(dataset['answers'].shape)
        dataset.close()
