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
        l = list(np.copy(self.board).flatten().tolist())
        for b, _, _ in self._history:
            l.extend(list(np.copy(b).flatten().tolist()))
        return hash(tuple(l))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def move(self, x, y, update_allowed=True):
        super().move(x, y, update_allowed)

    def is_terminal(self):
        return self.terminal or not np.any(np.multiply(self.legal_moves(), self.allow_predict()))

    def find_children(self, model):
        if self.terminal:
            return set()

        moves = self.predict(model)

        return {self.make_move(x, y, moves[x, y]) for x, y in zip(*moves.nonzero())}

    def find_random_child(self, model):
        moves = self.predict(model)
        rnd_idx = np.random.choice(np.arange(361), p=moves.flatten())
        x, y = np.where(np.arange(361).reshape((19, 19)) == rnd_idx)

        return self.make_move(int(x), int(y), moves[x, y])

    @property
    def history(self):
        return [(np.copy(board), turn, score) for board, turn, score in self._history]

    def copy(self):
        return TsumegoNode(self.board, self.turn, self.history, self.problem_type, self.score)

    def make_move(self, x, y, weights):
        new_board = self.copy()
        new_board.move(x, y)
        new_board.exploration_weight = weights
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

        return self.is_solved() or 0

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

    for i in range(32):
        tree.do_rollout(board)
        print('Rollout', i)
        print([(tree.Q[n], tree.N[n]) for n in tree.children[board] if tree.N[n]])

    x = board
    while True:
        print(x)
        try:
            x = tree.choose(x)
        except RuntimeError:
            break
