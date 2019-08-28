from model import create_model
from parser import parse_sgf_file, get_board_data, add_color_features
from utils import print_problem, print_problem_and_answer
from board import Board
import numpy as np

WEIGHTS_PATH = 'weights.h5'


def init_model():
    model = create_model()
    model.load_weights(WEIGHTS_PATH)
    return model


def sgf_to_problem(sgf_path: str):
    sgf = parse_sgf_file(sgf_path).data[0]
    board_data, sz = get_board_data(sgf.data[0])
    brd = Board(board_data)
    black_to_play, white_to_play = add_color_features(brd.board)

    return black_to_play, white_to_play


def solve(sgf_path: str):
    black_to_play, _ = sgf_to_problem(sgf_path)
    model = init_model()

    move = 'black'

    while True:

        answer = model.predict(np.array([black_to_play, ]))
        black_answer = np.array(answer[0] == np.max(answer[0]), dtype=int).reshape((19, 19))
        print_problem_and_answer(black_to_play, black_answer)

        black_to_play[0 if move == 'black' else 1] += black_answer

        brd = Board(black_to_play[:2])
        if brd.benson_groups()[3]:
            print('Some groups are alive')
            return

        black_to_play[4 if move == 'black' else 5] = np.ones((19, 19))
        black_to_play[4 if move == 'white' else 5] = np.zeros((19, 19))
        move = 'white' if move == 'black' else 'black'

    return black_answer


x = solve(f'data/cho_chikun_intermediate/prob0011.sgf')
