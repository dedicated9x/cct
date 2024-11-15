
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    def __repr__(self):
        return f"{self.val}"

from typing import Optional

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
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        retval = self.isValidBST_(root, None, None, is_root=True)
        return retval

    def isValidBST_(self, node: Optional[TreeNode], parent: TreeNode, is_on_right: bool, is_root: bool) -> bool:
        if node is None:
            return True

        # print(node.val)


        node_is_valid = False
        if is_root:
            node_is_valid = True
        else:
            if is_on_right:
                if node.val > parent.val:
                    node_is_valid = True
            else:
                if node.val < parent.val:
                    node_is_valid = True

        right_subtree_is_valid = self.isValidBST_(node.right, node, is_on_right=True, is_root=False)
        left_subtree_is_valid = self.isValidBST_(node.left, node, is_on_right=False, is_root=False)

        if right_subtree_is_valid and left_subtree_is_valid and node_is_valid:
            return True
        else:
            return False

# root = create_bst_from_flattened_list([5,1,4,None,None,3,6])
# print(Solution().isValidBST(root))
root = create_bst_from_flattened_list([5,4,6,None,None,3,7])
print(Solution().isValidBST(root))



