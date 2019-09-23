from collections import defaultdict

from keras.models import Model

from sgf_solver.board.tsumego import TsumegoBoard
from sgf_solver.solver.node import Node


class TreeSearch:

    def __init__(self, model: Model):
        self.children = dict()
        self.model = model

    def rollout(self, node: Node, times: int = 1):
        for i in range(times):
            print(f'\rRollout: {i}', end='')
            path = self._select(node)
            parent, leaf = path[-2:]
            reward = self._expand_and_evaluate(parent, leaf)
            self._backup(path, reward)

    def _select(self, node: Node):
        path = [None, ]
        while True:
            path.append(node)

            if node not in self.children or node.board.solved() is not None:
                return path

            node = node.next_child()

    def _expand_and_evaluate(self, parent: Node, leaf: Node):
        """Add root to children nodes and return reward"""
        leaf.evaluate(self.model)

        if parent:
            self.children[parent].add(leaf)

        if leaf not in self.children:
            self.children[leaf] = set()

        return leaf.reward()

    def _backup(self, path, reward):
        """Send the reward back up to the ancestors of the leaf"""
        for node in reversed(path[1:]):
            node.add_value(reward)
            reward = 1-reward


if __name__ == '__main__':
    from sgf_solver.model.model import create_model
    from sgf_solver.constants import ProblemClass
    from utils import get_problems, print_from_collection

    model = create_model()
    tree = TreeSearch(model)

    probs = get_problems()
    print_from_collection(probs, 55555)
    prob = probs['problems'][55555]
    board = TsumegoBoard(ProblemClass.LIVE, board=prob)
    node = Node(board)

    tree.rollout(node, 1600)
