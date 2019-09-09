from enum import Enum
from itertools import product
from typing import Dict, Tuple, FrozenSet, Set, Optional

import numpy as np

from sgf_solver.board import GoBoard, Location

Coord = Tuple[int, int]
Chain = FrozenSet[Coord]
Score = Dict[int, int]


class ProblemType(Enum):
    LIVE = 'live'
    KILL = 'kill'


class TsumegoBoard(GoBoard):
    def __init__(self, problem_type: ProblemType, stones: int = 0, **kwargs):
        super().__init__(**kwargs)
        self._type = problem_type
        self._stones = stones

    def _get_groups(self, color: Location) -> Set[Chain]:
        unexplored = np.array(self._board == color, dtype=int)
        groups = set()

        for x, y in product(range(19), range(19)):

            if unexplored[x, y]:
                group = self._get_group(x, y)
                groups.add(group)
                unexplored[tuple(zip(*group))] = 0

        return groups

    def _count_target_stones(self):
        stones = 0
        for group in self._get_groups(
                Location.WHITE if self._type == ProblemType.KILL else Location.BLACK):
            stones += len(group)

        return stones

    def _get_region(self, loc: Location, x0: int, y0: int) -> Chain:
        explored = set()
        unexplored = {(x0, y0)}

        while unexplored:
            x, y = unexplored.pop()
            unexplored |= {coord for p, coord in self._get_surrounding(x, y) if p != loc}

            explored.add((x, y))
            unexplored -= explored

        return frozenset(explored)

    def _get_regions(self, color: Location):
        unexplored = np.array(self._board != color, dtype=int)
        regions = set()
        for x, y in product(range(19), range(19)):

            if unexplored[x, y]:
                region = self._get_region(color, x, y)
                regions.add(region)
                unexplored[tuple(zip(*region))] = 0

        return regions

    def _is_chain_eye(self, loc: Location, group, region):
        surrounding = self._get_chain_surrounding(loc, group)
        return region.issubset(surrounding)

    def _is_chain_alive(self, loc: Location, group, regions):
        eyes = 0
        for region in regions:
            if self._is_chain_eye(loc, group, region):
                eyes += 1

        return eyes > 1

    def alive_groups(self, loc: Location) -> Set[Chain]:
        groups = self._get_groups(loc)
        regions = self._get_regions(loc)

        while True:
            alive, dead = set(), set()

            for group in groups:
                if self._is_chain_alive(loc, group, regions):
                    alive.add(group)
                else:
                    dead.add(group)

            if not dead:
                return alive

            for group in dead:
                for region in regions:
                    if self._is_chain_eye(loc, group, region):
                        regions.remove(region)

            groups = alive

    def solved(self) -> Optional[bool]:
        if self._type == ProblemType.LIVE:

            if self.alive_groups(Location.BLACK):
                return True

            if not self._stones:
                self._stones = self._count_target_stones()

            if self._score[Location.WHITE] >= self._stones:
                return False

        if self._type == ProblemType.KILL:

            if self.alive_groups(Location.WHITE):
                return False

            if not self._stones:
                self._stones = self._count_target_stones()

            if self._score[Location.BLACK] >= self._stones:
                return True

        return None


if __name__ == '__main__':
    prob = np.array([
        # [0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # [1, -1, -1, 0, 1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # [0, 1, -1, 0, -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # [0, 1, -1, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, -1, 0, 1, 0]
    ])
    board = TsumegoBoard(ProblemType.LIVE, board=prob, turn=Location.BLACK)

    print(board.solved())
    import time

    start = time.time()
    for _ in range(1000):
        board.alive_groups(Location.BLACK)
    end = time.time()

    print(board.alive_groups(Location.BLACK))
    print(end - start)
