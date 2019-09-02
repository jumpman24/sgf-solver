import h5py
import numpy as np


def get_problems():
    return h5py.File('problems_all.h5', 'r')


def print_problem_and_answer(problem, answers=None):
    size_x, size_y = problem.shape
    if answers is None:
        answers = np.zeros(problem.shape)

    board = ''

    for x in range(size_x):
        for y in range(size_y):
            if problem[x, y] == 1:
                board += '○ '
            elif problem[x, y] == -1:
                board += '● '
            elif answers[x, y] == 1:
                board += 'x '
            else:
                board += '. '
        board += '\n'

    print(board)


def print_from_collection(collection, number):
    return print_problem_and_answer(collection['problem'][number],
                                    collection['answers'][number])


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
