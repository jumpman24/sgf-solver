import math
from collections import defaultdict


class MCTS:
    "Monte Carlo tree searcher. First rollout the tree then choose a move."

    def __init__(self, model, exploration_weight=1):
        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        self.children = defaultdict(set)  # children of each node
        self.model = model
        self.exploration_weights = dict

    def choose(self, node):
        """Choose the best successor of node. (Choose a move in the game)"""
        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")

        if node not in self.children:
            return node.find_random_child(self.model)

        def score(n):
            if self.N[n] == 0:
                return float("-inf")  # avoid unseen moves
            return self.Q[n] / self.N[n]  # average reward
        return max(self.children[node], key=score)

    def do_rollout(self, node):
        """Make the tree one layer better. (Train for one iteration.)"""
        path = self._select(node)
        leaf = path[-1]
        self._expand(leaf)

        reward = self._simulate(leaf)
        self._backpropagate(path, reward)

    def _select(self, node):
        """Find an unexplored descendent of `node`"""
        path = []
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                # node is either unexplored or terminal
                return path
            unexplored = self.children[node] - self.children.keys()
            if unexplored:
                n = node.find_random_child(self.model)
                path.append(n)
                return path
            node = self._uct_select(node)  # descend a layer deeper

    def _expand(self, node):
        """Update the `children` dict with the children of `node`"""
        if node in self.children:
            return  # already expanded
        self.children[node] = node.find_children(self.model)

    def _simulate(self, node):
        """Returns the reward for a random simulation (to completion) of `node`"""
        invert_reward = True
        while True:
            if node.is_terminal():
                reward = node.reward()
                return -reward if invert_reward else reward
            node = node.find_random_child(self.model)
            invert_reward = not invert_reward

    def _backpropagate(self, path, reward):
        "Send the reward back up to the ancestors of the leaf"
        print(path)
        for node in reversed(path):
            print(self.N[node], self.Q[node], end=' ')
            self.N[node] += 1
            self.Q[node] += reward
            if reward == 1:
                print(node)
            print(self.N[node], self.Q[node])
            reward = -reward  # 1 for me is 0 for my enemy, and vice versa

    def _uct_select(self, node):
        "Select a child of node, balancing exploration & exploitation"

        # All children of node should already be expanded:
        assert all(n in self.children for n in self.children[node])

        log_N_vertex = math.log(self.N[node])

        def uct(n):
            "Upper confidence bound for trees"
            return self.Q[n] / self.N[n] + n.exploration_weight * math.sqrt(
                log_N_vertex / self.N[n]
            )

        return max(self.children[node], key=uct)
