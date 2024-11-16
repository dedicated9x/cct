
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

class Solution:
    def flatten(self, root: Optional[TreeNode]) -> None:
        if root is None:
            return

        if root.left is None and root.right is None:
            return

        list_nodes = self.get_list(root)

        for idx in range(1, len(list_nodes) - 1):
            list_nodes[idx].left = None
            list_nodes[idx].right = list_nodes[idx+1]

        list_nodes[0].left = None
        list_nodes[0].right = list_nodes[1]
        list_nodes[-1].left = None
        list_nodes[-1].right = None

        # self._print_list(list_nodes)

    def get_list(self, node: Optional[TreeNode]):
        if node.left is None and node.right is None:
            return [node]

        if node.left is not None:
            list_left = self.get_list(node.left)
        else:
            list_left = []

        if node.right is not None:
            list_right = self.get_list(node.right)
        else:
            list_right = []

        _list = [node] + list_left + list_right
        return _list

    def _print_list(self, list_nodes):
        list_tuples = []
        for node in list_nodes:
            if node.left is None:
                left = None
            else:
                left = node.left.val
            if node.right is None:
                right = None
            else:
                right = node.right.val
            tuple = (node.val, left, right)
            list_tuples.append(tuple)
        print(list_tuples)



root = create_bst_from_flattened_list([1,2,5,3,4,None,6])
print(Solution().flatten(root))
root = create_bst_from_flattened_list([0])
print(Solution().flatten(root))

