from typing import List

class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        self.list_sums = [0]
        running_sum = 0
        for elem in nums:
            running_sum += elem
            self.list_sums.append(running_sum)

        # for i in range(len(nums)):
        #     for j in range(len(nums)):
        #         if i <= j:
        #             sum2 = self.get_sum(i, j)
        #             print(i, j, nums[i:(j+1)], sum2)

        if sum(nums) < target:
            return 0

        idx_left = 0
        idx_right = 0
        min_lenght = len(nums)
        while idx_left < len(nums) -1:
            current_sum = self.get_sum(idx_left, idx_right)
            current_lenght = idx_right - idx_left + 1

            if current_sum >= target and current_lenght < min_lenght:
                min_lenght = current_lenght

            if idx_right == len(nums) - 1:
                idx_left += 1
            else:
                if current_sum < target:
                    idx_right += 1
                else:
                    idx_left += 1
        return min_lenght

    def get_sum(self, i, j):
        return self.list_sums[j + 1] - self.list_sums[i]



print(Solution().minSubArrayLen(target = 7, nums = [2,3,1,2,4,3]))