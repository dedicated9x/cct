import numpy as np

class Node:
    def __init__(self, max_children=None, last_diff=None, remainder=None):
        self.max_children = max_children
        self.remainder = remainder

        self.last_diff = last_diff

        self.children = []
        self.parent = None

    def __repr__(self):
        return f"{self.remainder} [{self.max_children}]"

def is_square(num) -> bool:
    return int(np.sqrt(num)) ** 2 == num

class Solution:
    def numSquares(self, n: int) -> int:
        if n <= 3:
            return n

        if is_square(n):
            return 1

        root = Node(max_children=n, remainder=n)
        nodes_layer0 = [root]
        list_layers = [nodes_layer0]

        depth = 0
        square_has_been_found = False
        # for idx in range(3):
        while not square_has_been_found:
            previous_layer = list_layers[-1]
            next_layer = []

            for parent in previous_layer:
                max_sqrt = int(np.sqrt(parent.remainder))
                possible_diffs = [e ** 2 for e in range(1, max_sqrt + 1) if e ** 2 <= parent.max_children]

                for diff in possible_diffs:
                    remainder = parent.remainder - diff
                    if is_square(remainder):
                        square_has_been_found = True
                    child = Node(max_children=diff, remainder=remainder)
                    next_layer.append(child)

            list_layers.append(next_layer)


        return len(list_layers)
        # TODO wyznaczyc possible diffs



print(Solution().numSquares(n=12))
