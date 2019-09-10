from enum import Enum, IntEnum

BOARD_SIZE = 19
BOARD_SHAPE = (BOARD_SIZE, BOARD_SIZE)


class Location(IntEnum):
    BLACK = 1
    EMPTY = 0
    WHITE = -1


class ProblemClass(Enum):
    LIVE = 'live'
    KILL = 'kill'
