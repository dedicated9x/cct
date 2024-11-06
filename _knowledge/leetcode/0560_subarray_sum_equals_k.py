from typing import List
import numpy as np

class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        left_idx = 0
        right_idx = 0
        nums = np.array(nums)
        count = 0
        while left_idx <= len(nums) - 1:
            _sum = nums[left_idx:(right_idx + 1)].sum()
            if _sum == k:
                count += 1

            print(left_idx, right_idx, _sum)

            right_idx += 1
            if right_idx > len(nums) - 1:
                left_idx += 1
                right_idx = left_idx

        return count




print(Solution().subarraySum(nums = [1,1,1], k = 2))
print(Solution().subarraySum(nums = [1,2,3], k = 3))