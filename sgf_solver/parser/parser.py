import os

import h5py
import numpy as np

from sgf_solver.annotations import (
    CoordType,
    PositionType,
    CollectionType,
    NodeListType,
    DataCollectionType,
    DatasetType,
)
from sgf_solver.board import GoBoard
from sgf_solver.constants import BOARD_SHAPE, PROBLEM_DATASET, PROBLEM_PATH
from sgf_solver.enums import Location
from sgf_solver.exceptions import ParserError
from sgf_solver.parser.sgflib import GameTree, SGFParser


class TsumegoParser:
    def __init__(self, path: str):
        self._path = path
        self._sgf = None
        self._trees = None
        self._position = None

    def _parse_file(self) -> GameTree:
        if not os.path.exists(self._path):
            raise ParserError(f"File not found: {self._path}")

        with open(filepath) as file:
            sgf_data = file.read()

        return SGFParser(sgf_data).parse()[0]

    @property
    def sgf(self) -> GameTree:
        if self._sgf is None:
            self._sgf = self._parse_file()
        return self._sgf

    @staticmethod
    def get_coord(sgf_coord: str) -> CoordType:
        if len(sgf_coord) != 2:
            raise ParserError("Wrong coordinates")

        return ord(sgf_coord[1]) - 97, ord(sgf_coord[0]) - 97

    def _get_position(self) -> PositionType:
        root_node = self.sgf[0]
        black = root_node.get('AB').data
        white = root_node.get('AW').data

        board = np.zeros(BOARD_SHAPE, dtype=int)

        for stone in black:
            coord = self.get_coord(stone)
            board[coord] = Location.BLACK

        for stone in white:
            coord = self.get_coord(stone)
            board[coord] = Location.WHITE

        return board

    @property
    def position(self) -> PositionType:
        if self._position is None:
            self._position = self._get_position()
        return self._position

    def _collect_trees(self, game_tree: GameTree) -> CollectionType:
        if not game_tree.variations:
            return [game_tree.data, ]

        trees = []
        common_data = game_tree.data
        for variation in game_tree.variations:
            nested_trees = self._collect_trees(variation)

            for nt in nested_trees:
                trees.append(common_data + nt)

        return trees

    @staticmethod
    def _sort_trees(trees: CollectionType) -> CollectionType:
        correct, wrong = [], []

        for tree in trees:
            comment = tree[-1].get('C')
            if comment.value.startswith('Correct'):
                correct.append(tree)
            elif comment.value.startswith('Wrong'):
                wrong.append(tree)

        return correct + wrong

    @property
    def trees(self) -> CollectionType:
        if self._trees is None:
            self._trees = self._sort_trees(self._collect_trees(self.sgf))
        return self._trees

    def parse_tree(self, tree: NodeListType) -> DataCollectionType:
        comment = tree[-1].get('C')

        if comment.value.startswith('Correct'):
            value = 1
        elif comment.value.startswith('Wrong'):
            value = 0
        else:
            return [], [], []

        board = GoBoard(self.position)

        tree_problems, tree_values, tree_answers = [], [], []
        color = 'B'
        for node in tree[1:]:
            coord = self.get_coord(node.get(color).value)

            if not coord:
                raise ParserError("No move in move node")

            # find the next move
            answer = np.zeros(BOARD_SHAPE, dtype=int)
            answer[coord] = 1

            # store problem, value and answers
            board_position = (board.board * board.turn).reshape((1, 19, 19))
            board_legal_moves = board.legal_moves.reshape((1, 19, 19))
            board_position = np.vstack((board_position, board_legal_moves))
            tree_problems.append(board_position)
            tree_answers.append(answer)
            tree_values.append(value)

            # make a move
            board.move(coord)
            value = 1 - value
            color = 'B' if color == 'W' else 'W'

        return tree_problems, tree_values, tree_answers

    @staticmethod
    def flip_transpose(old_problems, old_values, old_answers) -> DataCollectionType:
        new_boards, new_values, new_answers = [], [], []

        for board, value, answer in zip(old_problems, old_values, old_answers):
            new_boards.extend([
                np.array(board),
                np.flip(board, axis=(1,)),
                np.flip(board, axis=(2,)),
                np.flip(board, axis=(1, 2)),
                np.transpose(board, axes=(0, 2, 1)),
                np.transpose(np.flip(board, axis=(1,)), axes=(0, 2, 1)),
                np.transpose(np.flip(board, axis=(2,)), axes=(0, 2, 1)),
                np.transpose(np.flip(board, axis=(1, 2)), axes=(0, 2, 1)),
            ])
            new_values.extend([value, value, value, value, value, value, value, value])
            new_answers.extend([
                np.array(answer),
                np.flip(answer, axis=(0,)),
                np.flip(answer, axis=(1,)),
                np.flip(answer, axis=(0, 1)),
                np.transpose(answer),
                np.transpose(np.flip(answer, axis=(0,))),
                np.transpose(np.flip(answer, axis=(1,))),
                np.transpose(np.flip(answer, axis=(0, 1))),
            ])

        return new_boards, new_values, new_answers

    def get_dataset(self, extend: bool = True) -> DatasetType:
        sgf_problems, sgf_values, sgf_answers = [], [], []
        correct_hashes = set()

        for tree in self.trees:
            hashes = []
            tree_problems, tree_values, tree_answers = [], [], []

            for problem, value, answer in zip(*self.parse_tree(tree)):
                # merge another answer to existing problems
                problem_hash = hash(str(problem))

                if value == 1:
                    correct_hashes.add(problem_hash)
                elif problem_hash in correct_hashes:
                    continue

                if problem_hash in hashes:
                    tree_answers[hashes.index(problem_hash)] += answer
                    continue

                hashes.append(problem_hash)
                tree_problems.append(problem)
                tree_values.append(value)
                tree_answers.append(answer)

            if extend:
                tree_problems, tree_values, tree_answers = self.flip_transpose(
                    tree_problems, tree_values, tree_answers)

            sgf_problems.extend(tree_problems)
            sgf_values.extend(tree_values)
            sgf_answers.extend(tree_answers)

        return np.array(sgf_problems), np.array(sgf_values), np.array(sgf_answers)


if __name__ == '__main__':
    all_sgf_files = []
    for directory, _, files in os.walk(PROBLEM_PATH):
        for filename in files:
            if os.path.splitext(filename)[1] == '.sgf':
                all_sgf_files.append(os.path.join(directory, filename))

    all_problems, all_values, all_answers = [], [], []

    good_count, bad_count = 0, 0
    for filepath in sorted(all_sgf_files):
        parser = TsumegoParser(filepath)
        problems, values, answers = parser.get_dataset()

        all_problems.extend(problems)
        all_values.extend(values)
        all_answers.extend(answers)

        good_count += sum(values)
        bad_count += len(problems) - sum(values)
        print(f"\rProblems created: {good_count + bad_count:-6d}, "
              f"good: {good_count:-4d}, bad: {bad_count:-4d}", end='')
    print()

    with h5py.File(PROBLEM_DATASET, 'w') as dataset:
        dataset.create_dataset('problems', data=all_problems)
        dataset.create_dataset('values', data=all_values)
        dataset.create_dataset('answers', data=all_answers)
        dataset.close()
