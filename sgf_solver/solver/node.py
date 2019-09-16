from sgf_solver.board.tsumego import TsumegoBoard
from keras.models import Model

import math
import numpy as np


class TsumegoNode:
    def __init__(self, board: TsumegoBoard):
        self._visits = np.zeros(361, dtype=int)
        self._value = None
        self._policy = None
        self._children = np.zeros(361, dtype=TsumegoNode)
        self.board = board

    def __hash__(self):
        return hash(self.board)

    @property
    def W(self):
        return self._value

    @property
    def N(self):
        return np.sum(self._visits)

    @property
    def Q(self):
        return self.W / self.N

    def next_child(self):
        log_N_vertex = math.log(self.N)

        action_values = self._policy * math.sqrt(2 * log_N_vertex / (1 + self._visits))
        idx = self._visits.nonzero()
        action_values[idx] += self._policy[idx] / self._visits[idx]

        next_idx = int(np.argmax(action_values))

        if isinstance(self._children[next_idx], TsumegoNode):
            return self._children[next_idx]

        self.make_move(next_idx)

    def add_value(self, value):
        self._value += value

    def evaluate(self, model: Model):
        value, policy = model.predict([[[self.board.board * self.board.turn]]])
        self._value = value.item()
        self._policy = policy * self.board.legal_moves.flatten()

    def reward(self):
        solved = self.board.solved()
        if solved is True:
            return 1
        if solved is False:
            return -1
        return self.W

    def make_move(self, next_idx: int):
        board = self.board.copy()
        board.move(divmod(next_idx, 19))

        self._children[next_idx] = TsumegoNode(board)
        return self._children[next_idx]
