
def union_of_two_segments(left1, right1, left2, right2):
    left = max(left1, left2)
    right = min(right1, right2)
    if left > right:
        return 0
    else:
        return right - left

class Solution:
    def computeArea(self, ax1: int, ay1: int, ax2: int, ay2: int, bx1: int, by1: int, bx2: int, by2: int) -> int:
        width = union_of_two_segments(ax1, ax2, bx1, bx2)
        height = union_of_two_segments(ay1, ay2, by1, by2)
        intersection = width * height

        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)

        union = area_a + area_b - intersection

        return union


print(Solution().computeArea(ax1 = -3, ay1 = 0, ax2 = 3, ay2 = 4, bx1 = 0, by1 = -1, bx2 = 9, by2 = 2))