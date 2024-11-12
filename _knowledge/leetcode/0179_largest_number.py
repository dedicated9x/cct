from typing import List
import numpy as np

class Solution:
    def largestNumber(self, nums: List[int]) -> str:
        if len(nums) == 1:
            return str(nums[0])

        strings = [str(e) for e in nums]
        max_len = max([len(e) for e in strings])
        zero_padded = [int(e.ljust(max_len, '0')) for e in strings]
        idx_max = np.argmax(zero_padded)

        reduced_nums = nums[:idx_max] + nums[(idx_max + 1):]
        return str(nums[idx_max]) + self.largestNumber(reduced_nums)

print(Solution().largestNumber(nums = [3,30,91,32,5,92, 9, 9132]))
