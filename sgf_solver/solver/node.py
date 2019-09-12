from sgf_solver.board.tsumego import TsumegoBoard
from sgf_solver.constants import ProblemClass


class TsumegoNode(TsumegoBoard):
    def __init__(self, problem_type: ProblemClass, **kwargs):
        super().__init__(problem_type, **kwargs)
        self.W = 0
        self.N = 0
        self.P = 0
        self.terminal = self.solved() is not None

    @property
    def Q(self):
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

    def make_move(self, coord):
        new_board = self.copy()
        new_board.move(coord)
        return new_board

    def find_children(self):
        if self.terminal:
            return set()

        return {self.make_move(coord) for coord in zip(self.legal_moves.nonzero())}
