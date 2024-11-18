from typing import List
import numpy as np

def int_to_list_bool(k, lenght):
    as_str = str(bin(k)).split("b")[1].rjust(lenght, "0")
    as_list = [int(e) for e in list(as_str)]
    return as_list

def calculate_arr():
    int_min = 0
    int_max = 2 ** 9 - 1
    arr = list(range(int_min, int_max + 1))
    arr = [int_to_list_bool(e, lenght=10) for e in arr]
    arr = np.array(arr)
    return arr

def calculate_list_sums(arr):
    list_sums = []
    for idx, row in enumerate(arr):
        _sum = sum([e * idx for idx, e in enumerate(row)])
        list_sums.append(_sum)
    return list_sums

class Solution:
    arr = calculate_arr()
    list_sums = calculate_list_sums(arr)

    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        list_idxs = [idx for idx, e in enumerate(self.list_sums) if e == n]
        list_combinations = []
        for idx in list_idxs:
            combination = [j for j, e in enumerate(self.arr[idx]) if e == 1]
            list_combinations.append(combination)

        # Filter by k
        list_combinations = [e for e in list_combinations if len(e) == k]
        return list_combinations

print(Solution().combinationSum3(k = 3, n = 9))