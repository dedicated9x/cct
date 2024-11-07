from typing import List

class Solution:
    def rob(self, nums: List[int]) -> int:
        if len(nums) == 1:
            return nums[0]
        elif len(nums) == 2:
            return max(nums[0], nums[1])

        list_maxes = [None] * len(nums)
        list_maxes[0] = nums[0]
        list_maxes[1] = max(nums[0], nums[1])
        for idx in range(2, len(nums)):
            new_max = max(
                list_maxes[idx-1],
                nums[idx] + list_maxes[idx-2]
            )
            list_maxes[idx] = new_max

        return list_maxes[-1]

print(Solution().rob(nums = [2,7,9,3,1]))
