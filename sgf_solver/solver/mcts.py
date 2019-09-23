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
    import os
    from sgf_solver.model.model import create_model
    from utils import get_problems, print_from_collection
    from sgf_solver.constants import WEIGHTS_PATH
    model = create_model()

    if not os.path.exists(WEIGHTS_PATH):
        print(WEIGHTS_PATH)
        exit(1)

    model.load_weights(WEIGHTS_PATH)

    tree = TreeSearch(model)

    probs = get_problems()
    print_from_collection(probs, 1234)
    prob = probs['problems'][1234]
    board = TsumegoBoard(board=prob[0])
    print(board.problem)
    node = Node(board)

    tree.rollout(node, 200)

    node.show_answer()
