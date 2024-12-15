from typing import List, Tuple
import numpy as np

def get_neighbours_v2(i, j, m, n) -> List[Tuple[int, int]]:
    list_neighbours = [
        (i-1, j-1),
        (i-1, j),
        (i-1, j+1),
        (i, j-1),
        (i, j+1),
        (i+1, j-1),
        (i+1, j),
        (i+1, j+1),
    ]
    list_neighbours = [(i, j) for i, j in list_neighbours if 0 <= i <= m-1 and 0 <= j <= n-1]
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

        # Grid calculation
        arr[arr == 1] = -1

        arr_outer = np.zeros((m + 2, n + 2))
        arr_outer[1:m + 1, 1:n + 1] = arr

        arr_orig = arr
        arr = arr_outer

        list_to_trawerse = []
        for i in range(m+2):
            for j in range(n+2):
                if arr[i, j] == 0:
                    list_to_trawerse.append((i, j))

        while (arr == -1).any():
            next_list_to_trawerse = []
            for center in list_to_trawerse:
                i, j = center
                center_value = arr[i, j]
                list_neighbours = get_neighbours_v2(i, j, m+2, n+2)
                for neighbour in list_neighbours:
                    i1, j1 = neighbour
                    if arr[i1, j1] == -1:
                        arr[i1, j1] = center_value + 1
                        next_list_to_trawerse.append((i1, j1))
            list_to_trawerse = next_list_to_trawerse

        # Calculate max_are
        list_2x2_mins = []
        for i in range(1, m):
            for j in range(1, n):
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
#
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