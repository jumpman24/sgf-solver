import h5py

DATASET_PATH = 'data/cho_chikun_elementary.h5'


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


if __name__ == '__main__':
    data = h5py.File(DATASET_PATH, 'r')
    print_problem_and_answer(data['problem'][120], data['answers'][120])
    print(data['problem'][120][2:])
