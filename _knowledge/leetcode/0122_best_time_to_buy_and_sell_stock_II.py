from typing import List

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if len(prices) == 1:
            return 0
        prices = self.remove_adjacent_same_values(prices)
        if len(prices) == 1:
            return 0

        labels = [None] * len(prices)

        if prices[0] < prices[1]:
            labels[0] = "low"
        else:
            labels[0] = "high"

        if prices[-1] > prices[-2]:
            labels[-1] = "high"
        else:
            labels[-1] = "low"

        for idx in range(1, len(prices) - 1):
            if prices[idx] == max(prices[idx-1], prices[idx], prices[idx+1]):
                labels[idx] = "high"
            if prices[idx] == min(prices[idx-1], prices[idx], prices[idx+1]):
                labels[idx] = "low"

        price_label_pairs = [(p, l) for p, l in zip(prices, labels) if l is not  None]

        if price_label_pairs[0][1] == "high":
            price_label_pairs = price_label_pairs[1:]
        if price_label_pairs[-1][1] == "low":
            price_label_pairs = price_label_pairs[:-1]

        total_profit = 0
        for idx in range(0, len(price_label_pairs), 2):
            profit = price_label_pairs[idx+1][0] - price_label_pairs[idx][0]
            total_profit += profit

        return total_profit

    def remove_adjacent_same_values(self, _list: List[int]):
        _list = _list + [None]
        result = []
        for idx in range(len(_list) - 1):
            if _list[idx] != _list[idx+1]:
                result.append(_list[idx])
        return result

# print(Solution().remove_adjacent_same_values(_list = [7,1,5,5,3,6,4]))
# print(Solution().remove_adjacent_same_values(_list = [7,7,1,5,5,3,3,3,6,4,4]))
                                                      # * *   * * *   *

# print(Solution().maxProfit(prices = [7,1,5,5,3,6,4]))
print(Solution().maxProfit(prices = [1]))
print(Solution().maxProfit(prices = [3,3]))

