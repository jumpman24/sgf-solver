import h5py
import numpy as np
from sgf_solver.constants import PROBLEM_DATASET


class ConsoleColor:
    CEND = '\33[0m'
    CBOLD = '\33[1m'
    CITALIC = '\33[3m'
    CURL = '\33[4m'
    CBLINK = '\33[5m'
    CBLINK2 = '\33[6m'
    CSELECTED = '\33[7m'

    CBLACK = '\33[30m'
    CRED = '\33[31m'
    CGREEN = '\33[32m'
    CYELLOW = '\33[33m'
    CBLUE = '\33[34m'
    CVIOLET = '\33[35m'
    CBEIGE = '\33[36m'
    CWHITE = '\33[37m'

    CBLACKBG = '\33[40m'
    CREDBG = '\33[41m'
    CGREENBG = '\33[42m'
    CYELLOWBG = '\33[43m'
    CBLUEBG = '\33[44m'
    CVIOLETBG = '\33[45m'
    CBEIGEBG = '\33[46m'
    CWHITEBG = '\33[47m'

    CGREY = '\33[90m'
    CRED2 = '\33[91m'
    CGREEN2 = '\33[92m'
    CYELLOW2 = '\33[93m'
    CBLUE2 = '\33[94m'
    CVIOLET2 = '\33[95m'
    CBEIGE2 = '\33[96m'
    CWHITE2 = '\33[97m'

    CGREYBG = '\33[100m'
    CREDBG2 = '\33[101m'
    CGREENBG2 = '\33[102m'
    CYELLOWBG2 = '\33[103m'
    CBLUEBG2 = '\33[104m'
    CVIOLETBG2 = '\33[105m'
    CBEIGEBG2 = '\33[106m'
    CWHITEBG2 = '\33[107m'


def get_problems(extended=True):
    return h5py.File(PROBLEM_DATASET.format('big' if extended else 'small'), 'r')


def print_problem(problem, answers=None):
    if answers is None:
        answers = np.zeros((19, 19))

    board = ''

    for x in range(19):
        for y in range(19):
            if problem[x, y] == 1:
                board += ConsoleColor.CBLUE + '●'
            elif problem[x, y] == -1:
                board += ConsoleColor.CYELLOW + '●'
            elif answers[x, y] == 1:
                board += ConsoleColor.CRED + ConsoleColor.CBLINK + 'x'
            else:
                board += '.'
            board += ConsoleColor.CEND + ' '
        board += '\n'

    print(board)


def print_from_collection(collection, number):
    return print_problem(collection['problems'][number],
                         collection['answers'][number])


if __name__ == '__main__':
    problems = get_problems()
    print_from_collection(problems, 35)
