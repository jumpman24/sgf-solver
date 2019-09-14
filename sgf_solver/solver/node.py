from sgf_solver.board.tsumego import TsumegoBoard
from sgf_solver.annotations import CoordType
from sgf_solver.constants import ProblemClass, BOARD_SHAPE
from keras.models import Model


class TsumegoNode:
    def __init__(self, tsumego: TsumegoBoard, coord: CoordType):
        self._tsumego = tsumego
        self.W = 0
        self.N = 0
        self.P = 0
        self.coord = None
        self.value = 0
        self.visits = 0
        self.policy = None
        self.terminal = self.solved() is not None

    def evaluate(self, model: Model):
        value, policy = model.predict([[[self._board]]])
        self.value = value.item()
        self.policy = policy.reshape(BOARD_SHAPE) * self.legal_moves

    def reward(self):
        solved = self.solved()
        if solved is True:
            return 1
        if solved is False:
            return -1
        return 0

    @property
    def Q(self):
        if self.N == 0:
            return float('-inf')
        return self.W / self.N

    def set_value(self, value: float):
        self.P = value

    def add_visit(self, value: float):
        print(f"Adding {value:.2f} to node")
        self.W += value
        self.N += 1

    def copy(self):
        board, turn, score = self._state
        return TsumegoNode(self._type, board=board, turn=turn, score=score, history=self.history)

    def create_child(self, coord, policy_value):
        node = self.copy()
        node.P = policy_value
        node.move(coord)
        return node

    def find_children(self, model: Model):
        if self.terminal:
            return set()

        print('find_children')
        value, policy = model.predict([[[self._board]]])
        self.W += value
        policy = policy.reshape((19, 19))
        policy *= self.legal_moves
        print('policy')
        return {self.create_child(coord, policy[coord]) for coord in zip(*policy.nonzero())}
