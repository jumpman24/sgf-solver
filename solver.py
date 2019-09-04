from model import create_model
from board import TsumegoBoard
from utils import get_problems, print_from_collection
import numpy as np
from mcts import MCTS

WEIGHTS_PATH = 'weights.h5'


def init_model():
    model = create_model()
    model.load_weights(WEIGHTS_PATH)
    return model


class TsumegoNode(TsumegoBoard):
    def __hash__(self):
        return hash(''.join(self.board.astype(str).flatten().tolist()))

    def __eq__(self, other):
        return self.history == other.history and self.state == other.state

    def move(self, x, y, update_allowed=True):
        super().move(x, y, update_allowed)

    def is_terminal(self):
        return self.terminal or not np.any(np.multiply(self.legal_moves(), self.allow_predict()))

    def find_children(self, model):
        if self.terminal:
            return set()

        moves = self.predict(model)

        return {self.make_move(x, y) for x, y in zip(*moves.nonzero())}

    def find_random_child(self, model):
        moves = self.predict(model).flatten()
        rnd_idx = np.random.choice(np.arange(361), p=moves)
        x, y = np.where(np.arange(361).reshape((19, 19)) == rnd_idx)

        return self.make_move(int(x), int(y))

    @property
    def history(self):
        return [(np.copy(board), turn, score) for board, turn, score in self._history]

    def copy(self):
        return TsumegoNode(self.board, self.turn, self.history, self.problem_type)

    def make_move(self, x, y):
        new_board = self.copy()
        new_board.move(x, y)

        return new_board

    def allow_predict(self):
        x, y = self.board.nonzero()
        area = np.ones((19, 19))

        if min(x) > 9:
            area[:min(x)-1, :] = 0
        elif max(x) < 11:
            area[max(x)+1:, :] = 0

        if min(y) > 9:
            area[:, :min(y)-1] = 0
        elif max(y) < 11:
            area[:, max(y)+1:] = 0

        return np.multiply(area, self.allowed_moves)

    def reward(self):
        if not self.is_terminal():
            raise RuntimeError(f"reward called on nonterminal board")
        print(self.is_solved(), self.winner)
        if self.winner is True:
            # It's your turn and you've already won.
            return 1
        if self.winner is False:
            return 0  # Your opponent has just won.
        if self.winner is None:
            return 0.5  # Not decided

    def predict(self, model):
        predicted = model.predict([[[self.board * self.turn]]]).reshape((19, 19))
        predicted *= self.allow_predict()
        return predicted / np.sum(predicted)


if __name__ == '__main__':
    model = init_model()

    problems = get_problems()
    print_from_collection(problems, 500)
    prob = problems['problem'][500]

    tree = MCTS(model)
    board = TsumegoNode(prob, 1, [], 'kill')

    tree.do_rollout(board)

    print(tree.choose(board))
