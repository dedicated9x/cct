# Cool one

from typing import List

class Solution:
    def canJump(self, nums: List[int]) -> bool:
        if len(nums) == 1:
            return True

        if nums[0] == 0:
            return False

        nums[-1] = 1


        # last    - len(nums) - 1
        # last-1  - len(nums) - 2
        # last-2  - len(nums) - 3
        right_idx = 0
        while right_idx <= len(nums) - 3:
            right_idx += 1

            # print("right", nums[right_idx])

            # if (nums[right_idx + 1] != 0 and nums[right_idx] == 0) or (right_idx + 1 == len(nums) - 1):
            if (nums[right_idx + 1] != 0 and nums[right_idx] == 0):
                left_idx = right_idx
                while left_idx >= 0:
                    left_idx -= 1

                    if left_idx == -1:
                        return False

                    # print("right", nums[right_idx], "left", nums[left_idx])

                    target_distance = right_idx - left_idx + 1
                    length = nums[left_idx]
                    if length >= target_distance:
                        break
        return True

print(Solution().canJump(nums = [2,3,1,1,4]))
print(Solution().canJump(nums = [3,2,1,0,4]))
print(Solution().canJump(nums = [3,2,1,0, 4,5,5,0]))
print(Solution().canJump(nums = [3,0,  2,1,0,0,  2,3,1,1,4]))

print(Solution().canJump(nums = [
    8,2,4,4,4,9,5,2,5,8,8,0,
    8,6,9,1,1,6,3,5,1,2,6,6,0,
    4,8,6,0,
    3,2,8,7,6,5,1,7,0,
    3,4,8,3,5,9,0,
    4,0,
    1,0,
    5,9,2,0,
    7,0,
    2,1,0,
    8,2,5,1,2,3,9,7,4,7,0,0,
    1,8,5,6,7,5,1,9,9,3,5,0,
    7,  5]))

print(Solution().canJump(nums = [2,1,0,0]))
print(Solution().canJump(nums =   [1,2,3]))
# print(Solution().canJump(nums = [2,1,0,2]))
