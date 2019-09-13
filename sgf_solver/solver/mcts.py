import math
from collections import defaultdict

from keras.models import Model

from sgf_solver.solver.node import TsumegoNode


class TreeSearch:

    def __init__(self, model: Model):
        self.children = {}
        self.model = model

    def rollout(self, node, times: int = 1):
        for i in range(times):
            print('rollout')
            path = self._select(node)
            leaf = path[-1]
            self._expand(leaf)
            reward = self._evaluate(leaf)
            self._backup(path, reward)

    def _select_move(self, node: TsumegoNode):
        # All children of node should already be expanded:
        # assert all(n in self.children for n in self.children[node])

        log_N_vertex = math.log(node.N)

        def uct(n):
            """Upper confidence bound for trees"""
            return n.Q + n.P * math.sqrt(log_N_vertex / n.N)

        return max(self.children[node], key=uct)

    def _select(self, node: TsumegoNode):
        path = []
        while True:
            print('select')
            path.append(node)

            if node not in self.children or node.terminal:
                return path

            node = self._select_move(node)

    def _expand(self, node):
        """Update the `children` dict with the children of `node`"""
        if node in self.children:
            return  # already expanded
        print('expand')
        self.children[node] = node.find_children(self.model)

    def _evaluate(self, node):
        if node.terminal:
            return node.reward()
        print('evaluate')
        value, _ = self.model.predict([[[node.board]]])
        return value.item()

    def _backup(self, path, reward):
        """Send the reward back up to the ancestors of the leaf"""
        for node in reversed(path):
            node.add_visit(reward)
            reward = -reward

    def choose(self, node):
        max_n = 0
        max_node = None

        for n in self.children[node]:
            if n.N > max_n:
                max_n = n.N
                max_node = n

        return max_node


if __name__ == '__main__':
    from sgf_solver.model.model import create_model
    from sgf_solver.constants import ProblemClass
    from utils import get_problems, print_from_collection

    model = create_model()
    tree = TreeSearch(model)

    probs = get_problems()
    print_from_collection(probs, 55555)
    prob = probs['problems'][55555]
    node = TsumegoNode(ProblemClass.LIVE, board=prob)

    tree.rollout(node, 1600)
