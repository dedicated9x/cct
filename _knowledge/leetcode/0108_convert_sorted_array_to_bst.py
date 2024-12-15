"""
Nie robi rownych odnog
"""

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    def __repr__(self):
        return f"{self.val}"

from typing import List, Optional

class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        if len(nums) == 0:
            return None
        elif len(nums) == 1:
            return TreeNode(nums[0], None, None)
        elif len(nums) == 2:
            root = TreeNode(nums[0], None, None)
            right = TreeNode(nums[1], None, None)
            root.right = right
            return root
        elif len(nums) == 3:
            left = TreeNode(nums[0], None, None)
            root = TreeNode(nums[1], None, None)
            right = TreeNode(nums[2], None, None)
            root.left = left
            root.right = right
            return root


        root_idx = int(len(nums) / 2)
        list_left, root_val, list_right = nums[:root_idx], nums[root_idx], nums[(root_idx + 1):]

        root = TreeNode(root_val, None, None)
        tree_left = self.sortedArrayToBST(list_left)
        tree_right = self.sortedArrayToBST(list_right)
        root.right = tree_right
        root.left = tree_left
        return root



# print(Solution().sortedArrayToBST([-10,-3,0,5,9]))
# print(Solution().sortedArrayToBST([-10,-3,0,1,5,9]))

s2 = Solution().sortedArrayToBST([0,1,2,3,4,5])
a = 2