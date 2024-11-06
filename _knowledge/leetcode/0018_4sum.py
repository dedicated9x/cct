from  typing import List
from collections import Counter

class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        nums = sorted(nums)

        sums2pairs = {}
        n_nums = len(nums)
        for i in range(n_nums):
            for j in range(n_nums):
                if i >= j:
                    continue
                _sum = nums[i] + nums[j]
                if _sum not in sums2pairs:
                    sums2pairs[_sum] = []
                sums2pairs[_sum].append((nums[i], nums[j]))

        # Remove duplicates
        sums2pairs = {k: list(set(v)) for k, v in sums2pairs.items()}

        list_quadruplets = []
        for sum1, pairs1 in sums2pairs.items():
            sum2 = target - sum1
            if sum2 in sums2pairs:
                pairs2 = sums2pairs[sum2]
                for pair1 in pairs1:
                    for pair2 in pairs2:
                        quadruplet = tuple(sorted([pair1[0], pair1[1], pair2[0], pair2[1]]))
                        list_quadruplets.append(quadruplet)

        # Remove duplicates
        list_quadruplets = list(set(list_quadruplets))


        # Filter quadruplets
        target_counter = Counter(nums)
        filtered_quadruplets = [quad for quad in list_quadruplets if not (Counter(quad) - target_counter)]
        return [list(e) for e in filtered_quadruplets]


print(Solution().fourSum(nums = [1,0,-1,0,-2,2], target = 0))