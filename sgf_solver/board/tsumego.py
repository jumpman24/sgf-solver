from itertools import product
from typing import Tuple, Set, Optional

import numpy as np

from sgf_solver.annotations import ChainType, CoordType
from sgf_solver.board import GoBoard
from sgf_solver.enums import Location, ProblemClass


class TsumegoBoard(GoBoard):
    def __init__(self, problem: ProblemClass = None, stones: int = 0, **kwargs):
        super().__init__(**kwargs)
        self._problem = problem
        self._stones = stones

    def __hash__(self):
        return hash(str(self._board) + str(self.legal_moves))

    @property
    def board_data(self):
        return np.array([self.board * self.turn, self.legal_moves])

    @property
    def problem(self):
        if self._problem is None:
            bx, by = np.where(self.board == Location.BLACK)
            wx, wy = np.where(self.board == Location.WHITE)

            # calculate average distance from the corner
            black_dist = 10 - np.average([abs(10 - bx), abs(10 - by)])
            white_dist = 10 - np.average([abs(10 - wx), abs(10 - wy)])

            self._problem = ProblemClass.LIVE if black_dist < white_dist else ProblemClass.KILL

        return self._problem

    def copy(self):
        board, turn, score = self.state
        history = self.history
        return TsumegoBoard(self._problem, board=board, turn=turn, score=score, history=history)

    def _get_groups(self, color: Location) -> Set[ChainType]:
        unexplored = np.array(self._board == color, dtype=int)
        groups = set()

        for coord in product(range(19), range(19)):

            if unexplored[coord]:
                group = self._get_group(coord)
                groups.add(group)
                unexplored[tuple(zip(*group))] = 0

        return groups

    def _count_target_stones(self):
        stones = 0
        for group in self._get_groups(
                Location.WHITE if self._problem == ProblemClass.KILL else Location.BLACK):
            stones += len(group)

        return stones

    def _get_region(self, loc: Location, coord0: CoordType) -> ChainType:
        explored = set()
        unexplored = {coord0}

        while unexplored:
            coord = unexplored.pop()
            unexplored |= {coord for p, coord in self._get_surrounding(coord) if p != loc}

            explored.add(coord)
            unexplored -= explored

        return frozenset(explored)

    def _get_regions(self, color: Location):
        unexplored = np.array(self._board != color, dtype=int)
        regions = set()
        for coord in product(range(19), range(19)):

            if unexplored[coord]:
                region = self._get_region(color, coord)
                regions.add(region)
                unexplored[tuple(zip(*region))] = 0

        return regions

    def _is_chain_eye(self, loc: Location, group, region):
        surrounding = self._get_chain_surrounding(loc, group)
        return region.issubset(surrounding)

    def _chain_eyes(self, loc: Location, group, regions):
        eyes = set()
        for region in regions:
            if self._is_chain_eye(loc, group, region):
                eyes.add(region)

        return eyes

    def alive_groups(self, loc: Location) -> Tuple[Set[ChainType], Set[ChainType]]:
        groups = self._get_groups(loc)
        regions = self._get_regions(loc)

        if groups:
            while True:
                alive, dead, vitals = set(), set(), set()

                for group in groups:
                    eyes = self._chain_eyes(loc, group, regions)

                    if len(eyes) >= 2:
                        alive.add(group)
                        vitals |= eyes
                    else:
                        dead.add(group)

                if not dead:
                    return alive, vitals

                for group in dead:
                    for region in regions.copy():
                        if self._is_chain_eye(loc, group, region):
                            regions.remove(region)

                groups = alive

        return set(), set()

    def moves_to_consider(self):
        moves = self.legal_moves

        for loc in [Location.BLACK, Location.WHITE]:
            _, eyes = self.alive_groups(loc)
            for region in eyes:
                moves[tuple(zip(*region))] = 0

        return moves

    def solved(self) -> Optional[bool]:
        if self.problem == ProblemClass.LIVE:

            if self.alive_groups(Location.BLACK)[0]:
                return True

            if not self._stones:
                self._stones = self._count_target_stones()

            if self._score[Location.WHITE] >= self._stones:
                return False

        if self.problem == ProblemClass.KILL:

            if self.alive_groups(Location.WHITE)[0]:
                return False

            if not self._stones:
                self._stones = self._count_target_stones()

            if self._score[Location.BLACK] >= self._stones:
                return True

        return None
