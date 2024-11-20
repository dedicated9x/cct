from typing import List
import numpy as np

class Square:
    def __init__(self, topleft, size):
        self.topleft = topleft
        self.size = size

    def consist_ones_only(self, arr):
        top, left = self.topleft
        # subarray = arr[top:top+self.size, left:left+self.size]
        # retval = (subarray == 1).all()

        new_row = arr[top+self.size-1, left:left+self.size]
        new_col = arr[top:top+self.size, left+self.size-1]
        retval2 = (np.concatenate([new_row, new_col]) == 1).all()

        # assert retval == retval2

        return retval2

    def get_children(self):
        top, left = self.topleft
        list_children = [
            Square(topleft=(top, left), size=self.size + 1),
            # Square(topleft=(top - 1, left), size=self.size + 1),
            # Square(topleft=(top, left - 1), size=self.size + 1),
            # Square(topleft=(top - 1, left - 1), size=self.size + 1),
        ]
        return list_children

    def is_valid(self, m, n):
        top, left = self.topleft
        if 0 <= top <= top + self.size <= m and 0 <= left <= left + self.size <= n:
            return True
        else:
            return False

    def __hash__(self):
        # Use a tuple of the unique attributes for hashing
        return hash((self.topleft, self.size))

    def __eq__(self, other):
        # Compare attributes for equality
        return isinstance(other, Square) and self.topleft == other.topleft and self.size == other.size

    def __repr__(self):
        return f"{self.topleft}, size: {self.size}"

class Solution:
    def maximalSquare(self, matrix: List[List[str]]) -> int:
        arr = np.array(matrix).astype(int)
        m, n = arr.shape
        max_size = min(m, n)

        size_to_candidates = [[] for e in range(max_size + 1)]

        for i in range(m):
            for j in range(n):
                square = Square(topleft=(i, j), size=1)
                size_to_candidates[1].append(square)

        list_sizes = []
        for size, list_candidates in enumerate(size_to_candidates):
            # print(size)
            if size == 0:
                continue
            if len(list_candidates) == 0:
                continue
            for square in list_candidates:
                if square.consist_ones_only(arr):
                    list_sizes.append(square.size)
                    if size == max_size:
                        continue
                    list_children = square.get_children()
                    list_children = [e for e in list_children if e.is_valid(m, n)]
                    for child in list_children:
                        size_to_candidates[size+1].append(child)
            if size == max_size:
                continue
            size_to_candidates[size+1] = list(set(size_to_candidates[size+1]))

        if len(list_sizes) == 0:
            return 0
        else:
            return max(list_sizes) ** 2

from _knowledge.leetcode.data._0221_data import MATRIX
from _knowledge.leetcode.data._0221_data_v2 import MATRIX2

print(Solution().maximalSquare(matrix = [
    ["1","0","1","0","0"],
    ["1","0","1","1","1"],
    ["1","1","1","1","1"],
    ["1","0","0","1","0"]
]))

print(Solution().maximalSquare(matrix = [["0","1"],["1","0"]]))
print(Solution().maximalSquare(matrix = [["0"]]))
print(Solution().maximalSquare(matrix = [["1"]]))
print(Solution().maximalSquare(matrix = [["1","1"],["1","1"]]))
print(Solution().maximalSquare(matrix = MATRIX))
# print(Solution().maximalSquare(matrix = MATRIX2))

"""
4
1
0
1
4
100
"""