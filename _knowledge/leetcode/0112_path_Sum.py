
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
        # self.sum = None

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
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        root.__class__.sum = None

        root.sum = root.val
        # setattr(root, 'sum', root.val)

        d = deque()
        d.append(root)
        while d:
            next_node = d.popleft()

            if next_node.left is not None:
                next_node.left.sum = next_node.left.val + next_node.sum
                # setattr(next_node.left, 'sum', next_node.left.val + next_node.sum)
                d.append(next_node.left)

            if next_node.right is not None:
                next_node.right.sum = next_node.right.val + next_node.sum
                # setattr(next_node.right, 'sum', next_node.right.val + next_node.sum)
                d.append(next_node.right)

            if (next_node.left is None and next_node.right is None) and next_node.sum == targetSum:
                return True
        return False





root = create_bst_from_flattened_list([5,4,8,11,None,13,4,7,2,None,None, None,1])
print(Solution().hasPathSum(root, targetSum=22))

root = create_bst_from_flattened_list([1,2,3])
print(Solution().hasPathSum(root, targetSum=5))

