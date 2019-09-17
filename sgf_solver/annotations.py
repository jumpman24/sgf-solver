from typing import List, Dict, Tuple, FrozenSet
from sgf_solver.parser.sgflib import Node
from numpy import ndarray

CoordType = Tuple[int, int]
ChainType = FrozenSet[CoordType]
ScoreType = Dict[int, int]
PositionType = ndarray
HistoryType = List[Tuple[PositionType, int, ScoreType]]
NodeListType = List[Node]
CollectionType = List[NodeListType]
DataCollectionType = Tuple[List[ndarray], List[ndarray], List[ndarray]]
DatasetType = Tuple[ndarray, ndarray, ndarray]
