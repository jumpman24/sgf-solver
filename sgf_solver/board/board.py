from typing import List, Dict, Tuple

import numpy as np

BLACK = 1
EMPTY = 0
WHITE = -1


class CoordinateError(Exception):
    pass


class GoBoard:
    def __init__(self,
                 board: np.ndarray,
                 turn: int,
                 score: Dict[int, int] = None,
                 history: List[Tuple[np.ndarray, int, Dict[int, int]]] = None):
        self._board = np.array(board, copy=True, dtype=int)
        self._turn = turn
        self._score = score or {BLACK: 0, WHITE: 0}
        self._history = history or []

    @property
    def board(self):
        return np.copy(self._board)

    @property
    def turn(self):
        return self._turn

    @property
    def history(self):
        return self._history.copy()

    @property
    def score(self):
        return self._score.copy()

    @property
    def state(self):
        return self.board, self.turn, self.score

    def _push_history(self):
        self._history.append(self.state)

    def _pop_history(self):
        try:
            self._board, self._turn, self._score = self._history.pop()
        except IndexError:
            pass

    def _get_loc(self, x, y):
        if not(0 <= x < 19 and 0 <= y < 19):
            raise CoordinateError(f"Coordinate {x, y} is out of bounds")
        return self._board[x, y]

    def _get_surrounding(self, x0, y0):
        """ Get surrounding stones """
        coords = (
            (x0, y0 - 1),
            (x0 + 1, y0),
            (x0, y0 + 1),
            (x0 - 1, y0),
        )

        for x, y in coords:
            try:
                yield self._get_loc(x, y), (x, y)
            except CoordinateError:
                pass

    def _get_same_loc(self, loc, x0, y0, traversed):
        traversed.add((x0, y0))

        # TODO: deal with RecursionError
        for p, coord in self._get_surrounding(x0, y0):
            if p == loc and coord not in traversed:
                traversed |= self._get_same_loc(loc, *coord, traversed)

        return traversed

    def get_group(self, x, y):
        loc = self._get_loc(x, y)
        if loc is EMPTY:
            raise CoordinateError(f"Coordinate {x, y} is empty")

        group = np.zeros((19, 19), dtype=int)
        group_loc = self._get_same_loc(loc, x, y, set())
        group[tuple(zip(*group_loc))] = 1
        return group


x = np.array([
    [1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
])