import os
from typing import List

import h5py
import numpy as np

from board import Board
from sgflib import SGFParser, Node, GameTree

BLACK = 1
EMPTY = 0
WHITE = -1
SGF_DIRS = ['data/cho_chikun_elementary', 'data/cho_chikun_intermediate']


def get_pos(s):
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
        x, y = get_pos(stone)
        board[x, y] = 1

    for stone in white:
        x, y = get_pos(stone)
        board[x, y] = -1

    return board, sz


def get_raw_board(board: List[List]):
    return ''.join([''.join([pos for pos in line]) for line in board])


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


def get_label_value(tree: GameTree):
    prop_comment = tree[-1].get('C')
    if prop_comment:
        return 1 if prop_comment.value == 'Correct.' else 0
    return 0.5


def get_labels(tree_list: List, sz=19):
    board = [[0 for _ in range(sz)] for _ in range(sz)]

    for tree in tree_list:
        x, y = get_pos(tree[0].get('B').data[0])
        if not board[x][y]:
            board[x][y] = get_label_value(tree)

    return board


def parse_sgf_file(filepath: str):
    with open(filepath) as file:
        sgf_data = file.read()

    return SGFParser(sgf_data).parse()


def convert_board(board_data):
    converted_plane = np.zeros((1, 19, 19))
    converted_plane[0] = board_data

    return converted_plane


def generate_boards(board_data):
    generated = []
    for data in board_data:
        generated.extend([
            np.array(data),
            np.flip(data, axis=(0,)),
            np.flip(data, axis=(1,)),
            np.flip(data, axis=(0, 1)),
            np.transpose(data),
            np.transpose(np.flip(data, axis=(0,))),
            np.transpose(np.flip(data, axis=(1,))),
            np.transpose(np.flip(data, axis=(0, 1)))])

    return generated


def parse_correct_tree(board, tree):
    boards = []
    answers = []

    brd = Board(board)
    for move in tree:
        if move.get('B'):
            x, y = get_pos(move.get('B').value)
            cur_ans = np.zeros((19, 19))
            cur_ans[x, y] = 1
            boards.append(np.array(brd.board, copy=True))
            answers.append(np.array(cur_ans, copy=True))
            brd.move(x, y)

        elif move.get('W'):
            x, y = get_pos(move.get('W').value)
            brd.move(x, y)

    return np.array(boards), np.array(answers)


def parse_wrong_tree(board, tree):
    boards = []
    answers = []

    brd = Board(np.array(board, copy=True) * -1, Board.WHITE)
    for move in tree:
        if move.get('W'):
            x, y = get_pos(move.get('W').value)
            cur_ans = np.zeros((19, 19))
            cur_ans[x, y] = 1
            boards.append(np.array(brd.board, copy=True))
            answers.append(np.array(cur_ans, copy=True))
            brd.move(x, y)
        elif move.get('B'):
            x, y = get_pos(move.get('B').value)
            brd.move(x, y)

    return np.array(boards), np.array(answers)


def parse_trees(board, trees):
    all_boards = []
    all_answers = []
    for tree in trees:
        comment = tree[-1].get('C')
        if not comment:
            continue

        try:
            if comment.value.startswith('Correct'):
                boards, answers = parse_correct_tree(np.array(board, copy=True), tree)

            elif comment.value.startswith('Wrong'):
                boards, answers = parse_wrong_tree(np.array(board, copy=True), tree)
            else:
                continue

            all_boards.extend(boards)
            all_answers.extend(answers)
        except Exception as err:
            print("Exception", err)
            print(Board(board))

    if not all_boards:
        return [], []

    all_boards = np.array(all_boards)
    all_answers = np.array(all_answers)

    unique_boards, indices = np.unique(all_boards, axis=0, return_inverse=True)

    boards = []
    answers = []
    for i in range(unique_boards.shape[0]):
        boards.append(unique_boards[i])
        answer = np.array(
            np.sum(all_answers[indices == i].reshape(np.count_nonzero(indices == i), 19, 19),
                   axis=0) > 0,
            np.int0)
        answers.append(answer)

    print(f'{len(boards):3d} positions')
    return [np.array(boards), np.array(answers)]


if __name__ == '__main__':
    all_sgf_files = []
    for d in SGF_DIRS:
        for f in os.listdir(d):
            all_sgf_files.append(os.path.join(d, f))

    all_feature_data = []
    all_labels_data = []
    for sgf_path in sorted(all_sgf_files):
        if os.path.splitext(sgf_path)[1] != '.sgf':
            continue

        sgf = parse_sgf_file(sgf_path).data[0]
        board_data, sz = get_board_data(sgf.data[0])
        trees = get_trees(sgf)

        print(f'Parsing {sgf_path}...', end=' ')
        boards, answers = parse_trees(np.array(board_data, copy=True), trees)

        for b in boards:
            all_feature_data.append(b)

        for l in answers:
            all_labels_data.append(l)

    data_features = np.array(generate_boards(all_feature_data))
    data_labels = np.array(generate_boards(all_labels_data))

    # data_features = data_features.reshape((data_features.shape[0], 1, 19, 19))

    dataset = h5py.File('problems_all' + '.h5', 'w')
    dataset.create_dataset('problem', data=data_features)
    dataset.create_dataset('answers', data=data_labels)
    print(dataset['problem'].shape)
    dataset.close()
