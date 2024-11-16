import numpy as np

def int_to_32bit_binary(n):
    # For negative numbers, use two's complement by masking with 0xFFFFFFFF
    return format(n & 0xFFFFFFFF, '032b')

class Solution:
    def rangeBitwiseAnd(self, left: int, right: int) -> int:
        if left == right:
            return left

        left = int_to_32bit_binary(left)
        right = int_to_32bit_binary(right)
        for idx in range(len(left)):
            if left[idx] != right[idx]:
                prefix = left[:idx]
                break
                # prefix2 = right[:idx]
        prefix = prefix.ljust(32, '0')
        retval = int(prefix, 2)

        return retval


print(Solution().rangeBitwiseAnd(left=2, right=5))
print(Solution().rangeBitwiseAnd(left=5, right=7))
# print(Solution().rangeBitwiseAnd(left=256+25, right=256+35 + 32))
