from typing import List
import numpy as np

class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        total_cost = [g - c for g, c in zip(gas, cost)]
        _sum = sum(total_cost)

        arr = np.full((len(total_cost), len(total_cost)), None)

        first_row = np.array(total_cost).cumsum()
        arr[0, :] = first_row

        for idx in range(1, len(total_cost)):
            previous_row = arr[idx-1, :]
            previous_cost = total_cost[idx -1]
            next_row = (previous_row - previous_cost)[1:].tolist() + [_sum]
            arr[idx, :] = next_row

        good_indices = []
        for idx, row in enumerate(arr):
            if (row >= 0).all():
                good_indices.append(idx)

        if len(good_indices) == 0:
            return -1
        else:
            return good_indices[0]



print(Solution().canCompleteCircuit(gas = [1, 2, 3, 4, 5], cost = [3, 4, 5, 1, 2]))