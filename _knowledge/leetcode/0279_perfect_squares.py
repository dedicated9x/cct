import numpy as np

class Node:
    def __init__(self, max_children=None, last_diff=None, remainder=None):
        self.max_children = max_children
        self.remainder = remainder
        self.last_diff = last_diff

    def __repr__(self):
        return f"{self.remainder} [{self.max_children}]"

class Solution:
    def numSquares(self, n: int) -> int:
        root = Node(max_children=n, remainder=n)
        max_sqrt = int(np.sqrt(root.remainder))
        possible_diffs = [e ** 2 for e in range(1, max_sqrt + 1) if e ** 2 <= root.max_children]

        nodes_layer1 = []
        for diff in possible_diffs:
            new_node = Node(max_children=diff, remainder=root.remainder - diff)
            nodes_layer1.append(new_node)

        a = 2
        # TODO wyznaczyc possible diffs



print(Solution().numSquares(n=31))