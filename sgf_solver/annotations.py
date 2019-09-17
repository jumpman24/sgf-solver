from typing import List, Dict, Tuple, Set, FrozenSet, Union

from numpy import ndarray

from sgf_solver.enums import Location
from sgf_solver.parser.sgflib import Node

CoordType = Tuple[int, int]
ChainType = Union[FrozenSet[CoordType], Set[CoordType]]
LocatedCoordType = Tuple[Location, CoordType]
LocatedSurroundType = Set[LocatedCoordType]

ScoreType = Dict[int, int]
PositionType = ndarray
StateType = Tuple[ndarray, int, ScoreType]
HistoryType = List[StateType]
NodeListType = List[Node]
CollectionType = List[NodeListType]
DataCollectionType = Tuple[List[ndarray], List[ndarray], List[ndarray]]
DatasetType = Tuple[ndarray, ndarray, ndarray]
