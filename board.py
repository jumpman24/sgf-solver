import numpy as np


class Board:
    EMPTY = -1
    BLACK = 0
    WHITE = 1

    def __init__(self, board: np.ndarray, turn=BLACK):
        self.board = np.array(board, copy=True)
        self._turn = turn
        self._history = []
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
            print("Illegal move", x, y)
            print(self)
            print()
            raise Exception(f"Cannot make a move on ({x},{y})")

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
            raise Exception("Illegal move: suicide")

    def _check_for_ko(self):
        """ Check move for ko """
        if len(self._history) > 1 and np.array_equal(self.board, self._history[-2][0]):
            self._pop_history()
            raise Exception("Illegal move: ko")

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
        if x < 0 or y < 0:
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


if __name__ == '__main__':
    from sgflib import SGFParser
    from parser import get_board_data

    with open('data/prob0009.sgf', 'r') as f:
        sgf = SGFParser(f.read()).parse()

    b, _ = get_board_data(sgf.data[0].data[0])
    brd = Board(b)
    print(brd)
    print()
    brd.move(0, 1)
    print(brd)
    print()
    brd.move(0, 2)
    print(brd)
    print()
    brd.move(0, 3)
    print(brd)
    print()
    brd.move(0, 0)
    print(brd)
    print()
    brd.move(1, 0)
    print(brd)
    print()
    brd.move(2, 0)
    print(brd)
    print()
    brd.move(3, 0)
    print(brd)
    print()
    brd.move(4, 0)
    print(brd)
    print()
    brd.move(9, 9)
    print(brd)
    print()
    brd.move(0, 4)
    print(brd)
    print()
