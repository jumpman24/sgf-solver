import os
from typing import List

from sgflib import SGFParser, Node, Cursor, GameTree

EMPTY = 0
BLACK = 1
WHITE = 2
PATH_TO_DATA = 'data/'
PATH_TO_FEATURES = 'features.txt'
PATH_TO_LABELS = 'labels.txt'


def get_pos(s):
    if len(s) != 2:
        raise Exception("Wrong coordinates")
    x = ord(s[0]) - 97
    y = ord(s[1]) - 97
    return x, y


def get_board_data(node: Node):
    if 'SZ' in node:
        sz = int(node['SZ'].value)
    else:
        sz = 19

    black = node.get('AB').data
    white = node.get('AW').data

    board = [[EMPTY for _ in range(sz)] for _ in range(sz)]

    for stone in black:
        x, y = get_pos(stone)
        board[x][y] = BLACK

    for stone in white:
        x, y = get_pos(stone)
        board[x][y] = WHITE

    return board, sz


def get_raw_board(board: List[List]):
    return ''.join([''.join([pos for pos in line]) for line in board])


def get_tree_data(cursor: Cursor):
    if cursor.atEnd:
        return cursor.game_tree

    result = []

    for child in range(len(cursor.children)):
        cursor.next(child)
        next_node = get_tree_data(cursor)

        if cursor.atEnd:
            result.append(next_node)
        else:
            result.extend(next_node)

        cursor.previous()

    return result


def get_label_value(tree: GameTree):
    prop_comment = tree[-1].get('C')
    if prop_comment:
        return 1 if 'correct' in prop_comment.value.lower() else 0
    return 0.5


def get_labels(tree_list: List, sz=19):
    board = [[0 for _ in range(sz)] for _ in range(sz)]

    for tree in tree_list:
        x, y = get_pos(tree[0].get('B').data[0])
        board[x][y] = get_label_value(tree)

    return board


def parse_sgf_file(sgf_path: str):
    with open(os.path.join(PATH_TO_DATA, sgf_path)) as file:
        sgf_data = file.read()

    return SGFParser(sgf_data).parse().cursor()


def save_to_file(train_data: List[List[List]], out_path):
    raw_data = []

    for example in train_data:
        features = ''.join([''.join([str(x) for x in line]) for line in example])

        raw_data.append(features)

    with open(out_path, 'w+') as out:
        out.write('\n'.join(raw_data))


def print_board(data: List[List]):
    res = ''
    for line in data:
        for item in line:
            if item == BLACK:
                res += 'X '
            elif item == WHITE:
                res += 'O '
            else:
                res += '. '
        res += '\n'

    print(res)


if __name__ == '__main__':
    all_sgf_files = os.listdir(PATH_TO_DATA)

    all_feature_data = []
    all_labels_data = []
    for sgf_path in all_sgf_files:
        sgf = parse_sgf_file(sgf_path)
        board_data, sz = get_board_data(sgf.node)
        all_feature_data.append(board_data)
        tree = get_tree_data(sgf)
        all_labels_data.append(get_labels(tree, sz))

    save_to_file(all_feature_data, PATH_TO_FEATURES)
    save_to_file(all_labels_data, PATH_TO_LABELS)
