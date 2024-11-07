from typing import List

class Solution:
    def maxArea(self, height: List[int]) -> int:
        heights_sorted_desc = sorted(list(set(height)), reverse=True)
        max_height = max(height)

        exact_height2idxs = {}
        for idx, e in enumerate(height):
            exact_height2idxs.setdefault(e, []).append(idx)

        # Simplify
        exact_height2idxs = {k: self.simplify_list(v) for k, v in exact_height2idxs.items()}


        atleast_height2idxs = {k: [] for k, v in exact_height2idxs.items()}
        atleast_height2idxs[max_height] = exact_height2idxs[max_height]
        for idx_h, h in enumerate(heights_sorted_desc):
            if idx_h == 0:
                continue
            idx_atleast_h = exact_height2idxs[h] + atleast_height2idxs[heights_sorted_desc[idx_h-1]]

            # Simplify
            idx_atleast_h = self.simplify_list(idx_atleast_h)

            atleast_height2idxs[h] = idx_atleast_h

        area = max([k * (max(v) - min(v)) for k, v in atleast_height2idxs.items()])
        return area

    def simplify_list(self, _list: List[int]):
        return list(set([min(_list), max(_list)]))


print(Solution().maxArea(height = [1,8,6,2,5,4,8,3,7]))