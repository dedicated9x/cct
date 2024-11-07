from typing import List

class Solution:
    def rob(self, nums: List[int]) -> int:
        if len(nums) == 1:
            return nums[0]
        elif len(nums) == 2:
            return max(nums)

        return max(
            self.rob(nums[:-1]),
            nums[-1] + self.rob(nums[:-2])
        )

print(Solution().rob(nums = [2,7,9,3,1]))
