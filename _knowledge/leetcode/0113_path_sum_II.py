
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
        # self.sum = None

    def __repr__(self):
        return f"{self.val}, sum:{self.sum}"


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
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> List[List[int]]:
        if root is None:
            return []

        root.__class__.sum = None
        root.__class__.parent = None
        root.__class__.is_root = False

        root.sum = root.val
        root.is_root = True

        list_leaves = []

        d = deque()
        d.append(root)
        while d:
            next_node = d.popleft()

            if next_node.left is not None:
                next_node.left.parent = next_node
                next_node.left.sum = next_node.left.val + next_node.sum
                d.append(next_node.left)

            if next_node.right is not None:
                next_node.right.parent = next_node
                next_node.right.sum = next_node.right.val + next_node.sum
                d.append(next_node.right)

            if (next_node.left is None and next_node.right is None):
                list_leaves.append(next_node)

        list_paths = []
        for leaf in list_leaves:
            if leaf.sum == targetSum:
                path_leaf_to_root = self.get_path_leaf_to_root(leaf)
                path = list(reversed(path_leaf_to_root))
                list_paths.append(path)

        return list_paths

    def get_path_leaf_to_root(self, leaf: TreeNode) -> List[int]:
        list_nodes = []
        current_node = leaf
        while not current_node.is_root:
            list_nodes.append(current_node)
            current_node = current_node.parent
        list_nodes.append(current_node)
        list_values = [e.val for e in list_nodes]
        return list_values

root = create_bst_from_flattened_list([5,4,8,11,None,13,4,7,2,None,None,5,1])
print(Solution().pathSum(root, targetSum=22))

# root = create_bst_from_flattened_list([1,2,3])
# print(Solution().pathSum(root, targetSum=5))
#
# root = create_bst_from_flattened_list([1,2])
# print(Solution().pathSum(root, targetSum=0))

root = create_bst_from_flattened_list([2])
print(Solution().pathSum(root, targetSum=2))
