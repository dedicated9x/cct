from typing import List
from functools import cmp_to_key

# Define a comparator based on g
def comparator(x, y):
    xy = int(str(x) + str(y))
    yx = int(str(y) + str(x))

    if xy > yx:
        return 1  # f(x) > f(y)
    elif yx > xy:
        return -1  # f(y) > f(x)
    else:
        return 0  # f(x) == f(y)

# To sort the sequence `seq` based on `f`
def sort_by_function(seq):
    return sorted(seq, key=cmp_to_key(comparator), reverse=True)


class Solution:
    def largestNumber(self, nums: List[int]) -> str:
        if len(nums) == 1:
            return str(nums[0])

        sorted_seq = sort_by_function(nums)
        # print(sorted_seq)
        retval = "".join([str(e) for e in sorted_seq])
        retval = retval.lstrip("0")
        if retval == "":
            retval = "0"
        return retval

# print(Solution().largestNumber(nums = [3,30,91,32,5,92, 9, 9132]))
# print(Solution().largestNumber(nums = [3, 34, 3476]))
# print(Solution().largestNumber(nums = [4, 445, 4451]))
# print(Solution().largestNumber(nums = [4, 425, 4251]))
# print(Solution().largestNumber(nums = [4, 445, 4456]))
print(Solution().largestNumber(nums = [0, 0, 0]))
print(Solution().largestNumber(nums = [0, 4, 0]))
