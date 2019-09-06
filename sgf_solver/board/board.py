from typing import List, Dict, Tuple, Set

import numpy as np

BLACK = 1
EMPTY = 0
WHITE = -1

Coord = Tuple[int, int]
Chain = Set[Coord]
Score = Dict[int, int]


class CoordinateError(Exception):
    pass


class MoveError(Exception):
    pass


class GoBoard:
    def __init__(self, board: np.ndarray, turn: int, score: Dict[int, int] = None,
                 history: List[Tuple[np.ndarray, int, Score]] = None):
        self._board = np.array(board, copy=True, dtype=int)
        self._turn = turn
        self._score = score or {BLACK: 0, WHITE: 0}
        self._history = history.copy() or []

    @property
    def next_turn(self) -> int:
        return -self._turn

    def _flip_turn(self) -> None:
        self._turn = self.next_turn

    def _add_score(self, score) -> None:
        self._score[self._turn] += score

    @property
    def _state(self) -> Tuple[np.ndarray, int, Score]:
        return np.copy(self._board), self._turn, self._score.copy()

    def _push_history(self) -> None:
        self._history.append(self._state)

    def _pop_history(self) -> None:
        self._board, self._turn, self._score = self._history.pop()

    def _get_loc(self, x: int, y: int) -> int:
        if not (0 <= x < 19 and 0 <= y < 19):
            raise CoordinateError(f"Coordinate {x, y} is out of bounds")
        return self._board[x, y]

    def _get_surrounding(self, x0: int, y0: int) -> Tuple[int, Coord]:
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

    def _get_chain(self, loc: int, x0: int, y0: int) -> Chain:
        explored = set()
        unexplored = {(x0, y0)}

        while unexplored:
            x, y = unexplored.pop()
            unexplored |= {coord for p, coord in self._get_surrounding(x, y) if p == loc}

            explored.add((x, y))
            unexplored -= explored

        return explored

    def _get_group(self, x: int, y: int) -> Chain:
        loc = self._get_loc(x, y)
        if loc is EMPTY:
            raise CoordinateError(f"Coordinate {x, y} is empty")

        return self._get_chain(loc, x, y)

    def _get_group_liberties(self, x: int, y: int) -> Tuple[Chain, Chain]:
        group = self._get_group(x, y)
        liberties = set()

        for x, y in group:
            liberties |= {coord for p, coord in self._get_surrounding(x, y) if p == EMPTY}

        return group, liberties

    def _kill_group(self, x: int, y: int) -> int:
        group, liberties = self._get_group_liberties(x, y)

        if liberties:
            return 0

        self._board[tuple(zip(*group))] = 0
        return len(group)

    def _take_pieces(self, x0, y0):
        """ Remove pieces if needed """
        score = 0

        for p, (x, y) in self._get_surrounding(x0, y0):
            if p == self.next_turn:
                score += self._kill_group(x, y)

        self._add_score(score)

        return score

    def _check_suicide(self, x, y):
        group, liberties = self._get_group_liberties(x, y)

        if not liberties:
            self._pop_history()
            raise MoveError(f"Illegal move: suicide")

    def _check_for_ko(self):
        for history_board, turn, score in reversed(self._history):
            if np.array_equal(self._board, history_board):
                self._pop_history()
                raise MoveError(f"Illegal move: ko")

    def move(self, x, y):
        loc = self._get_loc(x, y)

        if loc != EMPTY:
            raise MoveError(f"Coordinate {x, y} is not empty.")


if __name__ == '__main__':
    import time

    stones = np.array([
        [1, -1, -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
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
    brd = GoBoard(stones, BLACK)
    start1 = time.time()
    for i in range(1000):
        gx = brd._get_group_liberties(0, 0)
    end1 = time.time()

    print(len(gx), end1 - start1)

    start2 = time.time()
    for i in range(1000):
        gy = brd._get_group_liberties(0, 2)
    end2 = time.time()

    print(len(gy), end2 - start2)

    start3 = time.time()
    for i in range(1000):
        gz = brd._get_group_liberties(2, 0)
    end3 = time.time()

    print(len(gz), end3 - start3)
