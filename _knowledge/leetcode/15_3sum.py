from typing import List
from collections import Counter

class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        list_results = []

        nums = sorted(nums)
        n_zeros = len([e for e in nums if e == 0])
        if n_zeros >= 3:
            list_results.append([0, 0, 0])
        if n_zeros > 1:
            nums = self.leave_one_zero(nums)

        count_dict = dict(Counter(nums))
        list_duplicates = [k for k,v in count_dict.items() if v >= 2]
        nums_as_dict = {k: None for k in nums}
        for num in list_duplicates:
            diff = 0 - 2 * num
            if diff in nums_as_dict:
                list_results.append([num, num, diff])
        # Remove duplicates
        nums = sorted(list(set(nums)))

        if len(nums) < 3:
            return list_results

        nums_as_dict = {k: None for k in nums}
        left_idx = 0
        right_idx = 1
        while left_idx < len(nums) - 1:
            left_val = nums[left_idx]
            right_val = nums[right_idx]
            diff = 0 - left_val - right_val

            if (diff in nums_as_dict) and (diff not in [left_val, right_val]):
                list_results.append([diff, left_val, right_val])

            right_idx += 1
            if right_idx == len(nums):
                left_idx += 1
                right_idx = left_idx + 1

        return list_results


    def leave_one_zero(self, nums):
        negatives = [e for e in nums if e < 0]
        positives = [e for e in nums if e > 0]
        return negatives + [0] + positives

# print(Solution().threeSum([-1,0,1,2,-1,-4]))
print(Solution().threeSum([-1,0,0,1,2,-1,-4]))