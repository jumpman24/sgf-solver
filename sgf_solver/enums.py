from enum import Enum, IntEnum


class Location(IntEnum):
    BLACK = 1
    EMPTY = 0
    WHITE = -1


class ProblemClass(Enum):
    LIVE = 'live'
    KILL = 'kill'
