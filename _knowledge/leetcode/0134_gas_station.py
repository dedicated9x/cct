from typing import List
import numpy as np

class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        total_cost = [g - c for g, c in zip(gas, cost)]
        _sum = sum(total_cost)

        if _sum < 0:
            return - 1

        good_indices = []

        previous_row = np.array(total_cost).cumsum()
        for idx in range(1, len(total_cost) + 1):
            print(previous_row)
            if (previous_row >= 0).all() == True:
                good_indices.append(idx - 1)
            # print(idx - 1, previous_row)
            if idx <= len(total_cost):
                previous_cost = total_cost[idx -1]
                next_row = np.array((previous_row - previous_cost)[1:].tolist() + [_sum])
                previous_row = next_row

        if len(good_indices) == 0:
            return -1
        else:
            return good_indices[0]



print(Solution().canCompleteCircuit(gas = [1, 2, 3, 4, 5], cost = [3, 4, 5, 1, 2]))