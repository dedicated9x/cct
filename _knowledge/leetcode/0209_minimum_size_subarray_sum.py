from typing import List

class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        list_sums = [0]
        running_sum = 0
        for elem in nums:
            running_sum += elem
            list_sums.append(running_sum)

        list_lenghts = []
        for i in range(len(list_sums)):
            for j in range(len(list_sums)):
                if i <= j:
                    _sum = list_sums[j] - list_sums[i]
                    if _sum >= target:
                        lenght = j - i
                        list_lenghts.append(lenght)

        if len(list_lenghts) >= 1:
            min_len = min(list_lenghts)
            return min_len
        else:
            return 0



print(Solution().minSubArrayLen(target = 7, nums = [2,3,1,2,4,3]))