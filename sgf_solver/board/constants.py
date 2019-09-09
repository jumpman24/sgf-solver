from enum import IntEnum
from typing import Dict, Tuple, FrozenSet

BOARD_SIZE = 19
BOARD_SHAPE = (BOARD_SIZE, BOARD_SIZE)

CoordType = Tuple[int, int]
ChainType = FrozenSet[CoordType]
ScoreType = Dict[int, int]


class Location(IntEnum):
    BLACK = 1
    EMPTY = 0
    WHITE = -1
