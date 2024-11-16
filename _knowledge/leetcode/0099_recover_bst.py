
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
    def recoverTree(self, root: Optional[TreeNode]) -> None:
        tree_as_list = self.flatten(root)

        down_idxs = []
        for idx in range(1, len(tree_as_list)):
            if tree_as_list[idx - 1].val > tree_as_list[idx].val:
                down_idxs.append(idx)

        if len(down_idxs) == 2:
            idx_left = down_idxs[0] - 1
            idx_right = down_idxs[1]
        elif len(down_idxs) == 1:
            idx_left = down_idxs[0] - 1
            idx_right = down_idxs[0]
        else:
            pass

        buff = tree_as_list[idx_right].val
        tree_as_list[idx_right].val = tree_as_list[idx_left].val
        tree_as_list[idx_left].val = buff

        # tree_as_list2 = self.flatten(root)
        # print(tree_as_list2)

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


root = create_bst_from_flattened_list([1,2,None])
print(Solution().recoverTree(root))
root = create_bst_from_flattened_list([1,3,None,None,2])
print(Solution().recoverTree(root))
root = create_bst_from_flattened_list([3,1,4,None,None,2])
print(Solution().recoverTree(root))
root = create_bst_from_flattened_list([2,4,1,None,None,3])
print(Solution().recoverTree(root))
"""
False
False
True
True
"""


