from itertools import product

import numpy as np

from sgf_solver.board import GoBoard
from sgf_solver.enums import Location
from utils import get_problems


class GroupStatus:
    ALIVE = 'alive'
    DEAD = 'dead'
    CRITICAL = 'critical'


class GroupAnalyzer(GoBoard):

    def _get_region(self, loc: Location, coord0):
        explored = set()
        unexplored = {coord0}

        while unexplored:
            coord = unexplored.pop()
            unexplored |= {coord for p, coord in self._get_adjacent(coord) if p != loc}

            explored.add(coord)
            unexplored -= explored

        return set(explored)

    def _get_regions(self, color: Location):
        unexplored = np.array(self._board != color, dtype=int)
        regions = []
        for coord in product(range(19), range(19)):

            if unexplored[coord]:
                region = self._get_region(color, coord)
                regions.append(region)
                unexplored[tuple(zip(*region))] = 0

        return regions

    def group(self, loc: Location):
        return np.array(self.board == loc, dtype=int)

    def get_filled_region(self, loc: Location):
        group = self.group(loc)

        top = np.count_nonzero(group[0, :]) > 0
        bottom = np.count_nonzero(group[-1, :]) > 0
        left = np.count_nonzero(group[:, 0]) > 0
        right = np.count_nonzero(group[:, -1]) > 0

        padded = np.pad(group, [1, 1], 'constant', constant_values=[(top, bottom), (left, right)])

        filled = np.maximum.accumulate(padded, 0) & \
                 np.maximum.accumulate(padded, 1) & \
                 np.maximum.accumulate(padded[::-1, :], 0)[::-1, :] & \
                 np.maximum.accumulate(padded[:, ::-1], 1)[:, ::-1]

        return set(zip(*filled[1:-1, 1:-1].nonzero()))

    def get_interiors(self, loc: Location):
        regions = self._get_regions(loc)
        filled_regions = self.get_filled_region(loc)

        interiors = []
        for region in regions:
            if region.issubset(filled_regions):
                interiors.append(region)

        return interiors


if __name__ == '__main__':
    problems = get_problems(True)

    prob = np.array([
        [0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    ])
    print(prob.shape)
    board = GroupAnalyzer(prob)

    print(board.get_interiors(Location.BLACK))
