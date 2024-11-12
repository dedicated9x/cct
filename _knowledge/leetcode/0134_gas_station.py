from typing import List
import numpy as np

class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        total_cost = [g - c for g, c in zip(gas, cost)]
        _sum = sum(total_cost)

        if _sum < 0:
            return - 1

        cumsum = np.cumsum(total_cost)

        min_index = cumsum.argmin() + 1

        if min_index == len(cumsum):
            min_index = 0

        return int(min_index)



print(Solution().canCompleteCircuit(gas = [1, 2, 3, 4, 5], cost = [3, 4, 5, 1, 2]))
print(Solution().canCompleteCircuit(gas = [5, 1, 2, 3, 4], cost = [2, 3, 4, 5, 1]))
print(Solution().canCompleteCircuit(gas = [4, 5, 1, 2, 3], cost = [1, 2, 3, 4, 5]))