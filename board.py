import numpy as np


class MoveException(Exception):
    pass


class Board:
    EMPTY = -1
    BLACK = 0
    WHITE = 1
    BLACK_LIBERTIES = 2
    WHITE_LIBERTIES = 3

    def __init__(self, stones: np.ndarray, turn=BLACK, history=None):
        self.board_size = stones.shape[1]
        self.board = np.array(stones, copy=True)

        self._turn = turn
        self._history = history or []
        self._score = {
            self.BLACK: 0,
            self.WHITE: 0,
        }

    def __str__(self):
        black = self.board[0]
        white = self.board[1]

        board = []

        for x in range(19):
            line = ''
            for y in range(19):
                if black[x, y] == 1:
                    line += '○ '
                elif white[x, y] == 1:
                    line += '● '
                else:
                    line += '. '
            board.append(line)

        return '\n'.join(board)

    def copy(self):
        return Board(self.board, self._turn, self._history)

    @property
    def turn(self):
        """ Current turn color """
        return 'Black' if self._turn is self.BLACK else 'White'

    @property
    def score(self):
        """ Current score """
        return {'Black': self._score[self.BLACK],
                'White': self._score[self.WHITE]}

    @property
    def _next_turn(self):
        """ Next turn """
        return self.WHITE if self._turn is self.BLACK else self.BLACK

    def move(self, x, y):
        """ Makes a move at given coordinates """
        loc = self._get_loc(x, y)
        if loc is not self.EMPTY:
            raise MoveException(f"Point [{x},{y}] is already occupied.")

        self._push_history()
        self.board[self._turn, x, y] = 1
        taken = self._take_pieces(x, y)

        if taken == 0:
            self._check_for_suicide(x, y)

        self._check_for_ko()
        self._flip_turn()

    def _check_for_suicide(self, x, y):
        """ Check move for suicide """
        if self.count_liberties(x, y) == 0:
            self._pop_history()
            raise MoveException(f"Point [{x},{y}] is a suicide.")

    def _check_for_ko(self):
        """ Check move for ko """
        if len(self._history) > 1 and np.array_equal(self.board, self._history[-2][0]):
            self._pop_history()
            raise MoveException("Illegal move: ko")

    def _take_pieces(self, x, y):
        """ Remove pieces if needed """
        score = 0
        for p, (x1, y1) in self._get_surrounding(x, y):
            liberties = self.count_liberties(x1, y1)
            if p is self._next_turn and liberties == 0:
                score += self._kill_group(x1, y1)

        self._add_score(score)
        return score

    def _flip_turn(self):
        """ Change current turn color """
        self._turn = self._next_turn
        return self._turn

    @property
    def _state(self):
        """ Return game state """
        return np.array(self.board, copy=True), self._turn, self._score

    def _load_state(self, state):
        """ Load given game state """
        self.board, self._turn, self._score = state

    def _push_history(self):
        """ Push game state to history """
        self._history.append(self._state)

    def _pop_history(self):
        """ Load and remove game state from history """
        try:
            state = self._history.pop()
            self._load_state(state)
        except IndexError:
            pass

    def _add_score(self, score):
        """ Add point to current player's score """
        self._score[self._turn] += score

    def _get_loc(self, x, y):
        """ Get location state """
        if x < 0 or y < 0 or x >= self.board_size or y >= self.board_size:
            return None

        elif self.board[0, x, y] == 1:
            return self.BLACK

        elif self.board[1, x, y] == 1:
            return self.WHITE

        else:
            return self.EMPTY

    def _get_surrounding(self, x, y):
        """ Get surrounding stones """
        coords = (
            (x, y - 1),
            (x + 1, y),
            (x, y + 1),
            (x - 1, y),
        )

        result = []
        for coord in coords:
            pos = self._get_loc(*coord)
            if pos is not None:
                result.append((pos, coord))

        return result

    def _get_group(self, x, y, traversed):
        """ Recursively get surrounding locations of the same color """
        loc = self._get_loc(x, y)

        locations = []
        for p, (a, b) in self._get_surrounding(x, y):
            if p is loc and (a, b) not in traversed:
                locations.append((p, (a, b)))

        traversed.add((x, y))

        if locations:
            return traversed.union(*[self._get_group(a, b, traversed) for _, (a, b) in locations])
        else:
            return traversed

    def get_group(self, x, y):
        """ Get group locations """
        if self._get_loc(x, y) not in [self.BLACK, self.WHITE]:
            raise Exception("No group to get")

        return self._get_group(x, y, set())

    def _kill_group(self, x, y):
        """ Remove group from the board and return stones amount """
        loc = self._get_loc(x, y)
        if loc is None or loc is self.EMPTY:
            raise Exception("Can only kill black or white group")

        group = self.get_group(x, y)
        score = len(group)

        for x1, y1 in group:
            self.board[:, x1, y1] = 0

        return score

    def _get_liberties(self, x, y, traversed):
        """ Recursively get liberties of group """
        loc = self._get_loc(x, y)

        if loc is self.EMPTY:
            return {(x, y)}

        else:
            locations = [
                (p, (a, b)) for p, (a, b) in self._get_surrounding(x, y)
                if (p is loc or p is self.EMPTY) and (a, b) not in traversed
            ]
            traversed.add((x, y))

            if locations:
                return set.union(*[
                    self._get_liberties(a, b, traversed)
                    for _, (a, b) in locations
                ])
            else:
                return set()

    def get_liberties(self, x, y):
        """ Get group liberties """
        return self._get_liberties(x, y, set())

    def count_liberties(self, x, y):
        """ Count group liberties """
        return len(self.get_liberties(x, y))

    def legal_moves(self):
        allowed = np.zeros((self.board_size, self.board_size))
        for x in range(self.board_size):
            for y in range(self.board_size):
                try:
                    self.move(x, y)
                    allowed[x, y] = 1
                    self._pop_history()
                except MoveException:
                    pass
        return allowed.astype(int)

    def get_groups(self):
        black_groups = []
        white_groups = []

        for x in range(self.board_size):
            for y in range(self.board_size):
                loc = self._get_loc(x, y)

                if loc == self.BLACK:
                    group = self.get_group(x, y)
                    if group not in black_groups:
                        black_groups.append(group)

                elif loc == self.WHITE:
                    group = self.get_group(x, y)
                    if group not in white_groups:
                        white_groups.append(group)

        return black_groups, white_groups

    def _get_region(self, x, y, color, traversed):
        """ Recursively get surrounding locations of the same color """
        locations = []
        surroundings = self._get_surrounding(x, y)

        for p, (a, b) in surroundings:
            if p != color and (a, b) not in traversed:
                locations.append((p, (a, b)))

        traversed.add((x, y))

        if locations:
            return traversed.union(*[self._get_region(a, b, color, traversed) for _, (a, b) in locations])
        else:
            return traversed

    def get_region(self, x, y, color):
        if self._get_loc(x, y) == color:
            raise Exception("Invalid region")

        return self._get_region(x, y, color, set())

    def get_regions(self):
        black_regions = []
        white_regions = []

        black_ignore = set()
        white_ignore = set()
        for x in range(self.board_size):
            for y in range(self.board_size):
                loc = self._get_loc(x, y)

                if (x, y) not in black_ignore and loc in [self.EMPTY, self.WHITE]:
                    region = self.get_region(x, y, self.BLACK)
                    if region not in black_regions:
                        black_regions.append(region)
                        black_ignore.update(region)

                if (x, y) not in white_ignore and loc in [self.EMPTY, self.BLACK]:
                    region = self.get_region(x, y, self.WHITE)
                    if region not in white_regions:
                        white_regions.append(region)
                        white_ignore.update(region)

        return black_regions, white_regions

    def _is_group_vital_region(self, group, region):
        liberties = self.get_liberties(*list(group)[0])
        intersected = region.intersection(liberties)

        return region == intersected

    def is_vital_region(self, region, groups):
        return any([self._is_group_vital_region(group, region) for group in groups])

    def pass_alive_groups(self, groups, regions):
        alive_groups = []
        not_alive_groups = []
        for group in groups:
            vital_count = 0
            for region in regions:
                vital_count += self._is_group_vital_region(group, region)

            if vital_count >= 2:
                alive_groups.append(group)
            else:
                not_alive_groups.append(group)

        return alive_groups, not_alive_groups

    def remove_regions(self, groups, regions):
        removing = []
        for region in regions:
            to_remain = True

            for group in groups:
                if region.intersection(self.get_liberties(*list(group)[0])):
                    to_remain = False

            if not to_remain:
                removing.append(region)

        return removing

    def vital_regions(self, regions, groups):
        remaining = []
        for region in regions:
            if self.is_vital_region(region, groups):
                remaining.append(region)

        return remaining

    def benson_groups(self):
        black_groups, white_groups = self.get_groups()
        black_regions, white_regions = self.get_regions()

        while True:
            black_groups, black_not_alive = self.pass_alive_groups(black_groups, black_regions)
            white_groups, white_not_alive = self.pass_alive_groups(white_groups, white_regions)

            if not any([black_not_alive, white_not_alive]):
                break

            for region in self.remove_regions(black_not_alive, black_regions):
                black_regions.remove(region)

            for region in self.remove_regions(white_not_alive, white_regions):
                white_regions.remove(region)

            black_regions = self.vital_regions(black_regions, black_groups)
            white_regions = self.vital_regions(white_regions, white_groups)

        return black_groups, white_groups


class TsumegoBoard:
    BLACK_TO_PLAY = 'black'
    WHITE_TO_PLAY = 'white'
    TO_LIVE = 'live'
    TO_KILL = 'kill'

    def __init__(self, board: Board, to_play=BLACK_TO_PLAY, target=TO_LIVE):
        self.board = board
        self.to_play = to_play
        self.target = target

    def move_and_copy(self, x, y):
        new_board = self.board.copy()
        new_board.move(x, y)
        return TsumegoBoard(new_board, self.to_play, self.target)

    def is_solved(self):
        black_groups, white_groups = self.board.get_groups()
        black_alive, white_alive = self.board.benson_groups()

        if self.target == self.TO_LIVE:
            if self.to_play == 'black' and black_alive or self.to_play == 'white' and white_alive:
                print(f"The {self.to_play} group is alive.")
                return True

            if self.to_play == 'black' and not black_groups or self.to_play == 'white' and not white_groups:
                print(f"The {self.to_play} group is dead.")
                return False

        if self.target == self.TO_KILL:
            if self.to_play == 'black' and white_alive or self.to_play == 'white' and black_alive:
                print(f"The {self.to_play} failed to kill.")
                return False

            if self.to_play == 'black' and not black_groups or self.to_play == 'white' and not white_groups:
                print(f"The {self.to_play} killed.")
                return True

        return None


if __name__ == '__main__':
    from sgflib import SGFParser
    from parser import get_board_data
    from utils import print_problem

    with open('data/test/prob0001.sgf', 'r') as f:
        sgf = SGFParser(f.read()).parse()

    b, _ = get_board_data(sgf.data[0].data[0])
