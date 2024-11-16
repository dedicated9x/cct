import numpy as np

class Solution:
    def countPrimes(self, n: int) -> int:
        if n == 0:
            return 0
        if n == 1:
            return 0

        max_divisor = int(np.sqrt(n))
        table = [1] * n
        table[0] = 0
        table[1] = 0

        for i in range(2, max_divisor + 1):
            if table[i] == 1:
                for j in range(2 * i, n, i):
                    table[j] = 0
        retval = sum(table)
        return retval


print(Solution().countPrimes(0))
print(Solution().countPrimes(1))
print(Solution().countPrimes(2))
print(Solution().countPrimes(3))
print(Solution().countPrimes(4))
print(Solution().countPrimes(5))
print("\n")
print(Solution().countPrimes(10))
print(Solution().countPrimes(32))