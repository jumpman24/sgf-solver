import math
from collections import defaultdict

import numpy as np
from keras.models import Model

from .node import TsumegoNode


class TreeSearch:

    def __init__(self, model: Model):
        self.W = defaultdict(int)
        self.N = defaultdict(int)
        self.P = defaultdict(int)
        self.children = {}
        self.model = model

    def rollout(self, node, times: int = 1):
        for _ in range(times):
            path = self._select(node)
            leaf = path[-1]
            self._expand(leaf)
            reward = self._evaluate(leaf)
            self._backup(path, reward)

    def _select(self, node: TsumegoNode):
        path = []
        while True:
            path.append(node)

            if node not in self.children or node.is_terminal:
                return path

            node = self._select_move(node)

    def _expand(self, node):
        """Update the `children` dict with the children of `node`"""
        if node in self.children:
            return  # already expanded

        self.children[node] = node.find_children(self.model)

    def _select_move(self, node: TsumegoNode):

        # All children of node should already be expanded:
        value, policy = self.model.predict(node)
        node.P = value.item()
        policy = policy.reshape((19, 19)) * node.legal_moves

        policy_max = np.max(policy)

        log_N_vertex = math.log(self.N[node])

        def uct(n):
            "Upper confidence bound for trees"
            return self.W[n] / self.N[n] + n.exploration_weight * math.sqrt(
                log_N_vertex / self.N[n]
            )

        return max(self.children[node], key=uct)
