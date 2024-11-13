from typing import Optional, List

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

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


class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if root is None:
            return []
        node_values = []
        level = 0
        self.levelOrder_(root, node_values, level)
        return node_values

    def levelOrder_(self, root: Optional[TreeNode], node_values: List[List[int]], level: int) -> bool:
        try:
            node_values[level].append(root.val)
        except IndexError:
            node_values.append([root.val])

        if root.left is not None:
            self.levelOrder_(root.left, node_values, level + 1)
        if root.right is not None:
            self.levelOrder_(root.right, node_values, level + 1)



root = create_bst_from_flattened_list([3,9,20,None,None,15,7])
print(Solution().levelOrder(root))
root = create_bst_from_flattened_list([1])
print(Solution().levelOrder(root))
root = create_bst_from_flattened_list([])
print(Solution().levelOrder(root))

