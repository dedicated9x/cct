
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    def __repr__(self):
        return f"{self.val}"


def create_bst_from_flattened_list(values):
    # Edge case: if list is empty, return None
    if not values:
        return None

    # Initialize the root of the tree
    root = TreeNode(values[0])
    queue = [root]
    i = 1  # Start with the second element in the list

    while i < len(values):
        current = queue.pop(0)  # Get the current node from the queue

        # Left child
        if i < len(values) and values[i] is not None:
            current.left = TreeNode(values[i])
            queue.append(current.left)
        i += 1

        # Right child
        if i < len(values) and values[i] is not None:
            current.right = TreeNode(values[i])
            queue.append(current.right)
        i += 1

    return root

from typing import Optional, List
from collections import deque

class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        tree_as_list = self.flatten(root)

        for idx in range(len(tree_as_list) - 1):
            if tree_as_list[idx].val >= tree_as_list[idx + 1].val:
                return False
        return True

    def flatten(self, root: Optional[TreeNode]) -> List[TreeNode]:
        tree_as_list = [root]
        d = deque()
        d.append(root)
        while d:
            next_node = d.popleft()

            if next_node.left is not None:
                idx = tree_as_list.index(next_node)
                tree_as_list = tree_as_list[:idx] + [next_node.left] + tree_as_list[idx:]
                d.append(next_node.left)

            if next_node.right is not None:
                idx = tree_as_list.index(next_node)
                tree_as_list = tree_as_list[:(idx + 1)] + [next_node.right] + tree_as_list[(idx + 1):]
                d.append(next_node.right)
        return tree_as_list

root = create_bst_from_flattened_list([5,1,4,None,None,3,6])
print(Solution().isValidBST(root))
root = create_bst_from_flattened_list([5,4,6,None,None,3,7])
print(Solution().isValidBST(root))
root = create_bst_from_flattened_list([5,4,7,None,None,6,8])
print(Solution().isValidBST(root))
root = create_bst_from_flattened_list([5])
print(Solution().isValidBST(root))
"""
False
False
True
True
"""


