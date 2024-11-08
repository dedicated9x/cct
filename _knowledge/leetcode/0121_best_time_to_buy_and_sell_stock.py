from typing import List

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        maxes = [None] * len(prices)
        mins = [None] * len(prices)

        maxes[0] = prices[0]
        mins[-1] = prices[-1]

        for idx in range(1, len(prices) - 1):
            next_max = max(maxes[idx-1], prices[idx])
            maxes[idx] = next_max

        for idx in range(len(prices) - 2, 0, -1):
            next_min = min(mins[idx+1], prices[idx])
            mins[idx] = next_min

        pairs = [(x, y) for x, y in zip(maxes[:-1], mins[1:])]

        profits = [e[1] - e[0] for e in pairs]
        retval = max(profits + [0])
        return retval

print(Solution().maxProfit(prices = [7,1,5,3,8,6,4]))
print(Solution().maxProfit(prices = [7,1,5,3,6,4]))
print(Solution().maxProfit(prices = [7,6,4,3,1]))