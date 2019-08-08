import h5py
import numpy as np
ALL_PATH = 'problems_all.h5'
ELEMENTARY_PATH = 'data/cho_chikun_elementary.h5'
INTERMEDIATE_PATH = 'data/cho_chikun_intermediate.h5'


def print_problem_and_answer(problem_data, answer_data):
    black = problem_data[0]
    white = problem_data[1]

    board = ''

    for x in range(19):
        for y in range(19):
            if black[x, y] == 1:
                board += '○ '
            elif white[x, y] == 1:
                board += '● '
            elif answer_data[x, y] == 1:
                board += 'x '
            else:
                board += '. '
        board += '\n'

    print("PROBLEM: ")
    print(board)


def print_problem(data):
    black = data[0]
    white = data[1]

    board = ''

    for x in range(19):
        for y in range(19):
            if black[x, y] == 1:
                board += '○ '
            elif white[x, y] == 1:
                board += '● '
            else:
                board += '. '
        board += '\n'

    print("PROBLEM: ")
    print(board)


def print_liberties(data):
    black = np.array(data[2], dtype=int)
    white = np.array(data[3], dtype=int)

    print(black + white)


def train_test_split(data):
    length = data['problem'].shape[0]
    threshold = int(length*0.8)
    # idx = np.random.permutation(length)
    # train_idx, test_idx = sorted(idx[:threshold]), sorted(idx[threshold:])

    train_x = data['problem'][:threshold]
    train_y = data['answers'][:threshold].reshape(threshold, -1)
    test_x = data['problem'][threshold:]
    test_y = data['answers'][threshold:].reshape(length-threshold, -1)

    return train_x, train_y, test_x, test_y


all_data = h5py.File(ALL_PATH, 'r')
# elementary_data = h5py.File(ELEMENTARY_PATH, 'r')
# intermediate_data = h5py.File(INTERMEDIATE_PATH, 'r')
