import numpy as np
from keras.models import Model

from sgf_solver.board.tsumego import TsumegoBoard


class Node:
    def __init__(self, board: TsumegoBoard):
        self._value = None
        self._policy = None
        self._children = np.zeros(361, dtype=Node)
        self.board = board

    def __hash__(self):
        return hash(str(self.board.board_data))

    @property
    def visits(self):
        return np.array([(child.N if child else 0) for child in self._children])

    @property
    def W(self):
        return self._value

    @property
    def N(self):
        return sum([child.N for child in self._children if child]) + 1

    @property
    def Q(self):
        return self.W / self.N

    def next_child(self):
        action_values = np.zeros(361)

        idx = self._children.nonzero()
        action_values[idx] = [n.Q for n in self._children[idx]]

        action_values += self._policy / (self.visits+1)

        next_idx = int(np.argmax(action_values))

        if isinstance(self._children[next_idx], Node):
            return self._children[next_idx]

        return self.make_move(next_idx)

    def add_value(self, value):
        self._value += value

    def evaluate(self, model: Model):
        value, policy = model.predict([[self.board.board_data]])
        self._value = value.item()
        self._policy = policy.flatten() * self.board.legal_moves.flatten()

    def reward(self):
        solved = self.board.solved()
        if solved is True:
            return 1
        if solved is False:
            return 0
        return self.W

    def make_move(self, next_idx: int):
        board = self.board.copy()
        board.move(divmod(next_idx, 19))

        self._children[next_idx] = Node(board)
        return self._children[next_idx]

    def perfect_variation(self):
        moves = []
        next_node = self

        while True:
            if next_node.N == 1:
                break

            idx = int(np.argmax(next_node.visits))
            moves.append(divmod(idx, 19))
            next_node = next_node._children[idx]

        return moves

    def show_answer(self):
        next_node = self

        while True:
            print(next_node.board)
            if next_node.N == 1:
                break

            idx = int(np.argmax(next_node.visits))
            next_node = next_node._children[idx]
