from typing import List
import numpy as np
import time


class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        left_idx = 0
        right_idx = 0
        nums = np.array(nums)

        cumsums = nums.cumsum()
        cumsums = np.array([0] + cumsums.tolist())

        count = 0
        while left_idx <= len(nums) - 1:
            _sum = cumsums[right_idx+1] - cumsums[left_idx]

            if _sum == k:
                count += 1

            # print(left_idx, right_idx, _sum)

            right_idx += 1
            if right_idx > len(nums) - 1:
                left_idx += 1
                right_idx = left_idx

        return count

    # def subarraySum2(self, nums: List[int], k: int) -> int:
    #     left_idx = 0
    #     right_idx = 0
    #     nums = np.array(nums)
    #     count = 0
    #     while left_idx <= len(nums) - 1:
    #         _sum = nums[left_idx:(right_idx + 1)].sum()
    #         if _sum == k:
    #             count += 1
    #
    #         # print(left_idx, right_idx, _sum)
    #
    #         right_idx += 1
    #         if right_idx > len(nums) - 1:
    #             left_idx += 1
    #             right_idx = left_idx
    #
    #     return count


from _knowledge.leetcode.data._0560_data import nums_large

print(Solution().subarraySum(nums = [1,1,1], k = 2))
print(Solution().subarraySum(nums = [1,2,3], k = 3))
print(Solution().subarraySum(nums = nums_large[:2000], k = -93))

