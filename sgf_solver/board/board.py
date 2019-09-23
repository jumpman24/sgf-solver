from itertools import product
from typing import Set

import numpy as np

from sgf_solver.annotations import (
    PositionType,
    ScoreType,
    StateType,
    HistoryType,
    CoordType,
    LocatedCoordType,
    LocatedSurroundType,
    ChainType,
)
from sgf_solver.constants import BOARD_SHAPE
from sgf_solver.enums import Location
from sgf_solver.exceptions import CoordinateError, IllegalMoveError


class GoBoard:
    def __init__(self, board: PositionType,
                 turn: Location = Location.BLACK,
                 score: ScoreType = None,
                 history: HistoryType = None):
        self._board = np.array(board, copy=True, dtype=int)
        self._turn = turn
        self._score = score or {Location.BLACK: 0, Location.WHITE: 0}
        self._history = history.copy() if history else []
        self._illegal = []

    def __repr__(self):
        return f"GoBoard: {len(self._history)} moves, {self.turn_color} to play"

    def __str__(self):
        print_map = {
            Location.BLACK: '○ ',
            Location.WHITE: '● ',
            Location.EMPTY: '. ',
        }
        board = ""
        for x in range(19):
            for y in range(19):
                coord = x, y
                board += print_map[self._get_loc(coord)]
            board += "\n"

        return board

    @property
    def board(self):
        return np.copy(self._board)

    @property
    def history(self):
        return self._history.copy()

    @property
    def turn(self):
        """ Current player """
        return self._turn

    @property
    def turn_color(self):
        """ Current player color """
        return 'black' if self._turn is Location.BLACK else 'white'

    @property
    def next_turn(self) -> int:
        """ Next player color """
        return Location.BLACK if self._turn is Location.WHITE else Location.WHITE

    def _flip_turn(self) -> None:
        """ Change turn """
        self._turn = self.next_turn

    def _add_score(self, score: int) -> None:
        """ Add captured stones to score """
        self._score[self._turn] += score

    @property
    def state(self) -> StateType:
        """ Current game state
        Represented as current board position, turn and score
        """
        return np.copy(self._board), self._turn, self._score.copy()

    def _push_history(self) -> None:
        """ Add current state to game history """
        self._history.append(self.state)

    def _pop_history(self) -> None:
        """ Load previous board position """
        self._board, self._turn, self._score = self._history.pop()

    def _get_loc(self, coord: CoordType) -> Location:
        """ Get location of coordinate """
        if not all([0 <= xy < 19 for xy in coord]):
            raise CoordinateError(f"Coordinate {coord} is out of bounds")
        return Location(self._board[coord])

    def _get_surrounding(self, coord0: CoordType) -> LocatedCoordType:
        """ Get surrounding locations if possible """
        x0, y0 = coord0
        coords = (
            (x0, y0 - 1),
            (x0 + 1, y0),
            (x0, y0 + 1),
            (x0 - 1, y0),
        )

        for coord in coords:
            try:
                yield self._get_loc(coord), coord
            except CoordinateError:
                pass

    def _get_chain(self, loc: Location, coord0: CoordType) -> ChainType:
        """ Get connected chain of stones or empty area

        :param loc: color to retrieve
        :param coord0: position to start
        :return:
        """
        explored = set()
        unexplored = {coord0}

        while unexplored:
            coord = unexplored.pop()
            unexplored |= {coord for p, coord in self._get_surrounding(coord) if p == loc}

            explored.add(coord)
            unexplored -= explored

        return frozenset(explored)

    def _get_chain_surrounding(self, loc: Location, chain: ChainType) -> LocatedSurroundType:
        surrounding = set()

        for coord in chain:
            surrounding |= {coord for p, coord in self._get_surrounding(coord) if p is not loc}

        return surrounding

    def _get_group(self, coord: CoordType) -> ChainType:
        loc = self._get_loc(coord)

        if loc is Location.EMPTY:
            raise CoordinateError(f"Empty")

        return self._get_chain(loc, coord)

    def _get_area(self, coord: CoordType) -> ChainType:
        loc = self._get_loc(coord)

        if loc is not Location.EMPTY:
            raise CoordinateError(f"Not empty")

        return self._get_chain(loc, coord)

    def _get_region(self, loc: Location, coord0: CoordType) -> ChainType:
        explored = set()
        unexplored = {coord0}

        while unexplored:
            coord = unexplored.pop()
            unexplored |= {coord for p, coord in self._get_surrounding(coord) if p != loc}

            explored.add(coord)
            unexplored -= explored

        return frozenset(explored)

    def _get_liberties(self, group: ChainType) -> Set[CoordType]:
        liberties = set()

        for coord in group:
            liberties |= {coord for p, coord in self._get_surrounding(coord) if p is Location.EMPTY}

        return liberties

    def _kill_group(self, group: ChainType) -> int:
        liberties = self._get_liberties(group)

        if liberties:
            return 0

        self._board[tuple(zip(*group))] = 0
        return len(group)

    def _capture(self, coord0: CoordType):
        """ Remove pieces if needed """
        score = 0

        for p, coord in self._get_surrounding(coord0):
            if p == self.next_turn:
                group = self._get_group(coord)
                score += self._kill_group(group)

        return score

    def _check_suicide(self, coord: CoordType):
        """ Verify that played stone has at least one liberty """

        group = self._get_group(coord)
        liberties = self._get_liberties(group)

        if not liberties:
            self._pop_history()
            raise IllegalMoveError("Suicide")

    def _check_ko(self):
        """ Verify that super ko rule is not violated """
        for history_board, turn, score in reversed(self._history):
            if np.array_equal(self._board, history_board):
                self._pop_history()
                raise IllegalMoveError("Ko")

    def _set_illegal(self, coord0):
        self._illegal = []

        surroundings = self._get_surrounding(coord0)
        illegal = []
        moves_to_try = []
        for p, coord in surroundings:
            if p == Location.EMPTY:
                moves_to_try.append(coord)
            elif p == self.turn:
                group = self._get_group(coord)
                liberties = self._get_liberties(group)

                if len(liberties) == 1:
                    moves_to_try.append(liberties.pop())

        for coord in moves_to_try:
            try:
                self.move(coord, False)
                self._pop_history()
            except IllegalMoveError:
                illegal.append(coord)

        self._illegal = illegal

    def move(self, coord: CoordType, check: bool = True):
        if coord in self._illegal:
            raise IllegalMoveError("Ko")
        loc = self._get_loc(coord)

        if loc is not Location.EMPTY:
            raise IllegalMoveError("Not empty")

        self._push_history()
        self._board[coord] = self._turn
        captured = self._capture(coord)

        if captured:
            print(captured)
            self._add_score(captured)
        else:
            self._check_suicide(coord)

        self._check_ko()
        self._flip_turn()

        if check:
            self._set_illegal(coord)

    def make_pass(self):
        self._push_history()
        self._flip_turn()

    @property
    def legal_moves(self):
        legal_moves = np.ones(BOARD_SHAPE, dtype=int)
        legal_moves[self.board.nonzero()] = 0
        if self._illegal:
            legal_moves[tuple(zip(*self._illegal))] = 0
        return legal_moves
