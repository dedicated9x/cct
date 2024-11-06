from typing import Optional

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
    visited_nodes = {}
    def findTarget(self, root: Optional[TreeNode], k: int) -> bool:
        if root is None:
            return False
        else:
            diff = k - root.val
            if diff in self.visited_nodes:
                retval = True
            else:
                self.visited_nodes[root.val] = None
                retval = self.findTarget(root.left, k) or self.findTarget(root.right, k)
            print(root.val, retval)
            return retval



root = create_bst_from_flattened_list([5,3,6,2,4,None,7])
# print(Solution().findTarget(root, k=9))
# print(Solution().findTarget(root, k=28))

root = create_bst_from_flattened_list([1])
print(Solution().findTarget(root, k=2))

