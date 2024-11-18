from typing import List
import numpy as np
from collections import deque

class Square:
    def __init__(self, topleft, size):
        self.topleft = topleft
        self.size = size

    def consist_ones_only(self, arr):
        top, left = self.topleft
        subarray = arr[top:top+self.size, left:left+self.size]
        return (subarray == 1).all()

    def get_children(self):
        top, left = self.topleft
        list_children = [
            Square(topleft=(top, left), size=self.size + 1),
            Square(topleft=(top - 1, left), size=self.size + 1),
            Square(topleft=(top, left - 1), size=self.size + 1),
            Square(topleft=(top - 1, left - 1), size=self.size + 1),
        ]
        return list_children

    def is_valid(self, m, n):
        top, left = self.topleft
        if 0 <= top <= top + self.size <= m and 0 <= left <= left + self.size <= n:
            return True
        else:
            return False

    def __repr__(self):
        return f"{self.topleft}, size: {self.size}"

class Solution:
    def maximalSquare(self, matrix: List[List[str]]) -> int:
        arr = np.array(matrix).astype(int)
        m, n = arr.shape

        queue = deque()

        for i in range(m):
            for j in range(n):
                square = Square(topleft=(i, j), size=1)
                queue.append(square)

        list_sizes = []
        while queue:
            square = queue.popleft()
            if square.consist_ones_only(arr):
                list_sizes.append(square.size)
                list_children = square.get_children()
                list_children = [e for e in list_children if e.is_valid(m, n)]
                for child in list_children:
                    queue.append(child)

            if 4 in list_sizes:
                a = 2
            print(max(list_sizes))

        if len(list_sizes) == 0:
            return 0
        else:
            return max(list_sizes) ** 2

from _knowledge.leetcode.data._0221_data import MATRIX

# print(Solution().maximalSquare(matrix = [
#     ["1","0","1","0","0"],
#     ["1","0","1","1","1"],
#     ["1","1","1","1","1"],
#     ["1","0","0","1","0"]
# ]))
#
# print(Solution().maximalSquare(matrix = [["0","1"],["1","0"]]))
# print(Solution().maximalSquare(matrix = [["0"]]))
# print(Solution().maximalSquare(matrix = [["1"]]))
# print(Solution().maximalSquare(matrix = [["1","1"],["1","1"]]))
print(Solution().maximalSquare(matrix = MATRIX))