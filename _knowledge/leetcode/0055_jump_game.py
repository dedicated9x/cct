from typing import List

class Solution:
    def canJump(self, nums: List[int]) -> bool:
        if nums[0] == 0:
            if len(nums) == 1:
                return True
            else:
                return False

        left_idx = 0
        right_idx = 0

        list_segments = []
        while right_idx <= len(nums) - 3:
            right_idx += 1
            if nums[right_idx+1] != 0 and nums[right_idx] == 0:
                list_segments.append(nums[left_idx:(right_idx+1)])
                left_idx = right_idx + 1
                right_idx = left_idx

        list_segments.append(nums[left_idx:(right_idx+1)])
        list_bools = [self.is_segment_doable(segment) for segment in list_segments]

        print(list_segments)
        print(list_bools)

        if False in list_bools:
            return False
        else:
            return True

    def is_segment_doable(self, segment: List[int]) -> bool:
        if 0 not in segment:
            return True

        for idx, lenght in enumerate(segment):
            target_distance = len(segment) - idx
            if lenght >= target_distance:
                return True

        return False

# print(Solution().canJump(nums = [2,3,1,1,4]))
# print(Solution().canJump(nums = [3,2,1,0,4]))
# print(Solution().canJump(nums = [3,2,1,0, 4,5,5,0]))
# print(Solution().canJump(nums = [3,0,  2,1,0,0,  2,3,1,1,4]))

# print(Solution().canJump(nums = [0,2,3]))
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