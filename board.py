import numpy as np


class MoveException(Exception):
    pass


class Board:
    BLACK = 1
    EMPTY = 0
    WHITE = -1

    def __init__(self, stones: np.ndarray, turn=BLACK, history=None):
        self.board_size = stones.shape[1]
        self.board = np.array(stones, dtype=int, copy=True)

        self._turn = self.BLACK if turn == 0 else self.WHITE
        self._history = history or []
        self._score = {
            self.BLACK: 0,
            self.WHITE: 0,
        }

    def __str__(self):
        board = []

        for x in range(19):
            line = ''
            for y in range(19):
                if self.board[x, y] == self.BLACK:
                    line += '○ '
                elif self.board[x, y] == self.WHITE:
                    line += '● '
                else:
                    line += '. '
            board.append(line)

        return '\n'.join(board)

    @property
    def turn(self):
        """ Current turn color """
        return self._turn

    @property
    def score(self):
        """ Current score """
        return {'Black': self._score[self.BLACK],
                'White': self._score[self.WHITE]}

    @property
    def _next_turn(self):
        """ Next turn """
        return self.WHITE if self._turn == self.BLACK else self.BLACK

    def move(self, x, y):
        """ Makes a move at given coordinates """
        loc = self._get_loc(x, y)

        if loc != self.EMPTY:
            raise MoveException(f"Point [{x},{y}] is already occupied.")

        self._push_history()
        self.board[x, y] = self._turn
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
            if p == self._next_turn and liberties == 0:
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

        return self.board[x, y]

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
            if p == loc and (a, b) not in traversed:
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
        if loc is None or loc == self.EMPTY:
            raise Exception("Can only kill black or white group")

        group = self.get_group(x, y)
        score = len(group)

        for x1, y1 in group:
            self.board[x1, y1] = 0

        return score

    def _get_liberties(self, x, y, traversed):
        """ Recursively get liberties of group """
        loc = self._get_loc(x, y)

        if loc == self.EMPTY:
            return {(x, y)}

        else:
            locations = [
                (p, (a, b)) for p, (a, b) in self._get_surrounding(x, y)
                if (p == loc or p == self.EMPTY) and (a, b) not in traversed
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


class TsumegoBoard(Board):
    """It is always black to play!"""
    TO_LIVE = 'live'
    TO_KILL = 'kill'

    def __init__(self, stones, turn, history, problem_type=TO_LIVE, winner=None):
        super().__init__(stones, turn, history)
        self.problem_type = problem_type
        self.winner = winner

    @property
    def terminal(self):
        return self.winner is not None

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

    def make_move(self, x, y):
        new_board = TsumegoBoard(self.board.copy(),
                                 self.turn,
                                 self._history.copy(),
                                 self.problem_type)
        new_board.move(x, y)
        new_board.winner = new_board.is_solved()

        return new_board

    def is_solved(self):
        black_groups, white_groups = self.get_groups()
        black_alive, white_alive = self.benson_groups()

        if self.problem_type == self.TO_LIVE:
            if black_alive:
                print(f"Correct. The black group is alive.")
                return True

            if not black_groups:
                print(f"Wrong. The black group is dead.")
                return False

        if self.problem_type == self.TO_KILL:
            if white_alive:
                print(f"Wrong. The white group is alive.")
                return False

            if not white_groups:
                print(f"Correct. The white group is dead.")
                return True

        return None

    def reward(self):
        if not self.terminal:
            raise RuntimeError(f"reward called on nonterminal board")
        if self.winner is True:
            # It's your turn and you've already won. Should be impossible.
            return 1
        if self.winner is False:
            return 0  # Your opponent has just won. Bad.
        if self.winner is None:
            return 0.5  # Board is a tie
