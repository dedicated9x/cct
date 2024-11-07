from typing import List
# from  collections import Counter

class Solution:
    def maxArea(self, height: List[int]) -> int:
        list_heights = sorted(list(set(height)), reverse=True)
        max_height = max(list_heights)

        height2idxs = {}
        for idx, e in enumerate(height):
            height2idxs.setdefault(e, []).append(idx)


        height2idxs_v2 = {k: [] for k, v in height2idxs.items()}
        height2idxs_v2[max_height] = height2idxs[max_height]
        for idx_h, h in enumerate(list_heights):
            if idx_h == 0:
                continue
            new_value = height2idxs[h] + height2idxs_v2[list_heights[idx_h-1]]
            height2idxs_v2[h] = new_value
            a = 2
        a = 2


print(Solution().maxArea(height = [1,8,6,2,5,4,8,3,7]))