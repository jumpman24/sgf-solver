from enum import IntEnum
from typing import List, Dict, Tuple, Set, FrozenSet
from itertools import product
import numpy as np


Coord = Tuple[int, int]
Chain = FrozenSet[Coord]
Score = Dict[int, int]


class Location(IntEnum):
    BLACK = 1
    EMPTY = 0
    WHITE = -1


class CoordinateError(Exception):
    pass


class IllegalMoveError(Exception):
    pass


class GoBoard:
    def __init__(self, board: np.ndarray, turn: Location, score: Dict[int, int] = None,
                 history: List[Tuple[np.ndarray, int, Score]] = None):
        self._board = np.array(board, copy=True, dtype=int)
        self._turn = turn
        self._score = score or {Location.BLACK: 0, Location.WHITE: 0}
        self._history = history.copy() if history else []

    def __repr__(self):
        return f"GoBoard: {len(self._history)} moves, {self.turn} to play"

    def __str__(self):
        print_map = {
            Location.BLACK: '○ ',
            Location.WHITE: '● ',
            Location.EMPTY: '. ',
        }
        board = ""
        for x in range(19):
            for y in range(19):
                board += print_map[self._get_loc(x, y)]
            board += "\n"

        return board

    @property
    def turn(self):
        return 'black' if self._turn is Location.BLACK else 'white'

    @property
    def next_turn(self) -> int:
        return Location.BLACK if self._turn is Location.WHITE else Location.WHITE

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

    def _get_loc(self, x: int, y: int) -> Location:
        if not (0 <= x < 19 and 0 <= y < 19):
            raise CoordinateError(f"Coordinate {x, y} is out of bounds")
        return Location(self._board[x, y])

    def _get_surrounding(self, x0: int, y0: int) -> Tuple[Location, Coord]:
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

    def _get_chain(self, loc: Location, x0: int, y0: int) -> Chain:
        explored = set()
        unexplored = {(x0, y0)}

        while unexplored:
            x, y = unexplored.pop()
            unexplored |= {coord for p, coord in self._get_surrounding(x, y) if p == loc}

            explored.add((x, y))
            unexplored -= explored

        return frozenset(explored)

    def _get_group(self, x: int, y: int) -> Chain:
        loc = self._get_loc(x, y)

        if loc is Location.EMPTY:
            raise CoordinateError(f"Empty")

        return self._get_chain(loc, x, y)

    def _get_area(self, x: int, y: int) -> Chain:
        loc = self._get_loc(x, y)

        if loc is not Location.EMPTY:
            raise CoordinateError(f"Not empty")

        return self._get_chain(loc, x, y)

    def _get_group_liberties(self, x: int, y: int) -> Tuple[Chain, Set]:
        group = self._get_group(x, y)
        liberties = set()

        for x, y in group:
            liberties |= {coord for p, coord in self._get_surrounding(x, y) if p is Location.EMPTY}

        return group, liberties

    def _get_chain_surrounding(self, loc: Location, chain: Chain) -> Set[Tuple[Location, Coord]]:
        surrounding = set()

        for x, y in chain:
            surrounding |= {coord for p, coord in self._get_surrounding(x, y) if p is not loc}

        return surrounding

    def _kill_group(self, x: int, y: int) -> int:
        group, liberties = self._get_group_liberties(x, y)

        if liberties:
            return 0

        self._board[tuple(zip(*group))] = 0
        return len(group)

    def _capture(self, x0, y0):
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
            raise IllegalMoveError("Suicide")

    def _check_ko(self):
        for history_board, turn, score in reversed(self._history):
            if np.array_equal(self._board, history_board):
                self._pop_history()
                raise IllegalMoveError("Ko")

    def move(self, x, y):
        loc = self._get_loc(x, y)

        if loc is not Location.EMPTY:
            raise IllegalMoveError("Not empty")

        self._push_history()
        self._board[x, y] = self._turn
        captured = self._capture(x, y)

        if not captured:
            self._check_suicide(x, y)

        self._check_ko()
        self._flip_turn()

    @property
    def legal_moves(self):
        legal_moves = np.zeros((19, 19))
        for x, y in product(range(19), range(19)):
            try:
                self.move(x, y)
                self._pop_history()
                legal_moves[x, y] = 1
            except IllegalMoveError:
                pass

        return legal_moves
