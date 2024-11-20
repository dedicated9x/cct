from typing import List, Tuple
import numpy as np

def get_border_idx(arr: np.ndarray) -> List[Tuple[int, int]]:
    m, n = arr.shape
    list_idxs = []
    for i in range(m):
        for j in range(n):
            if i in [0, m-1] or j in [0, n-1]:
                list_idxs.append((i, j))
    return list_idxs

def get_neighbours(i, j, arr: np.ndarray) -> List[int]:
    list_neighbours = [
        arr[i-1, j-1],
        arr[i-1, j],
        arr[i-1, j+1],
        arr[i, j-1],
        arr[i, j+1],
        arr[i+1, j-1],
        arr[i+1, j],
        arr[i+1, j+1],
    ]
    return list_neighbours

class Solution:
    def maximalSquare(self, matrix: List[List[str]]) -> int:
        arr = np.array(matrix).astype(int)
        m, n = arr.shape
        max_size = min(m, n)

        if max_size == 1:
            if (arr == 1).any():
                return 1
            else:
                return 0

        arr[arr == 1] = -1
        list_border_idxs = get_border_idx(arr)
        for location in list_border_idxs:
            i, j = location
            if arr[i, j] == -1:
                arr[i, j] = 1

        last_max_value = 0
        while (arr == -1).any():
            for i in range(1, m-1):
                for j in range(1, n-1):
                    if arr[i, j] == -1:
                        list_neighbours = get_neighbours(i, j, arr)
                        if last_max_value in list_neighbours:
                            arr[i, j] = last_max_value + 1
            last_max_value += 1

        list_2x2_mins = []
        for i in range(0, m - 1):
            for j in range(0, n - 1):
                square_2x2 = [
                    arr[i, j],
                    arr[i+1, j],
                    arr[i, j+1],
                    arr[i+1, j+1],
                ]
                min_2x2 = min(square_2x2)
                list_2x2_mins.append(min_2x2)

        max_1x1 = arr.max()
        max_2x2 = max(list_2x2_mins)

        max_side_1x1 = 2 * (max_1x1 - 1) + 1
        max_side_2x2 = 2 * max_2x2

        max_side = max(max_side_1x1, max_side_2x2)
        max_area = max_side ** 2
        return int(max_area)



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
print(Solution().maximalSquare(matrix = MATRIX2))

"""
4
1
0
1
4
100
"""