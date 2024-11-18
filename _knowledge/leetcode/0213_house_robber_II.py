from typing import List

class Solution:
    def rob(self, nums: List[int]) -> int:
        if len(nums) <= 3:
            return max(nums)

        val1 = nums[0] + self.rob_not_circle(nums[2:-1])
        val2 = self.rob_not_circle(nums[1:])
        retval = max(val1, val2)

        return retval

    def rob_not_circle(self, nums: List[int]):
        if len(nums) <= 2:
            return max(nums)

        list_maxes = [None] * len(nums)
        list_maxes[0] = nums[0]
        list_maxes[1] = max(nums[0], nums[1])
        for idx in range(2, len(list_maxes)):
            next_max = max(
                nums[idx] + list_maxes[idx-2],
                list_maxes[idx-1]
            )

            list_maxes[idx] = next_max
        return list_maxes[-1]

# print(Solution().rob(nums = [1,2,3,1,5,6,7,2,4]))
print(Solution().rob(nums = [1,2,3,1]))
# print(Solution().rob(nums = [1,1,3,6,7,10,7,1,8,5,9,1,4,4,3]))