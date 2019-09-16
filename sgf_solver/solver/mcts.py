import math
from collections import defaultdict
from keras.models import Model
from sgf_solver.board.tsumego import TsumegoBoard
from sgf_solver.solver.node import TsumegoNode


class TreeSearch:

    def __init__(self, model: Model):
        self.children = defaultdict(set)
        self.model = model

    def rollout(self, node: TsumegoNode, times: int = 1):
        for i in range(times):
            print('rollout', i)
            path = self._select(node)
            parent, leaf = path[-2:]
            reward = self._expand_and_evaluate(parent, leaf)
            self._backup(path, reward)

    def _select(self, node: TsumegoNode):
        path = [None, ]
        while True:
            print('select')
            path.append(node)

            if node not in self.children or node.terminal:
                return path

            node = node.next_child()

    def _expand_and_evaluate(self, parent: TsumegoNode, leaf: TsumegoNode):
        """Add root to children nodes and return reward"""
        print('expand')
        if leaf not in self.children[node]:
            leaf.evaluate(self.model)

            if parent:
                self.children[parent].add(leaf)

        return leaf.reward()

    def _backup(self, path, reward):
        """Send the reward back up to the ancestors of the leaf"""
        for node in reversed(path):
            node.add_visit(reward)
            reward = -reward


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
    node = TsumegoNode(board)

    tree.rollout(node, 1600)
