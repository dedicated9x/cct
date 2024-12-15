# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

    def __repr__(self):
        return f"{self.val}"

def list_to_linkedlist(lst):
    if not lst:  # Handle empty list case
        return None
    head = ListNode(lst[0])  # Create the head node
    current = head
    for value in lst[1:]:
        current.next = ListNode(value)
        current = current.next
    return head


from typing import Optional

def insert(node: ListNode, head: ListNode) -> ListNode:
    # Case 1: insert at the beginning
    if node.val <= head.val:
        node.next = head
        return node
    else:
        ptr_before = head
        while ptr_before.next is not None:
            ptr_after = ptr_before.next
            # Case 2: insert in the middle
            if node.val <= ptr_after.val:
                ptr_before.next = node
                node.next = ptr_after
                return head
            ptr_before = ptr_after

        # Case 3: insert at the end
        ptr_before.next = node
        return head

class Solution:
    def insertionSortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if head is None:
            return None
        if head.next is None:
            return head

        ptr_queue = head.next
        ptr_retval = head
        ptr_retval.next = None

        while ptr_queue.next is not None:
            next_element = ptr_queue
            ptr_queue = ptr_queue.next

            # Detaching
            next_element.next = None

            # Insertion
            ptr_retval = insert(next_element, ptr_retval)

        ptr_retval = insert(ptr_queue, ptr_retval)
        return ptr_retval

_f = list_to_linkedlist

# print(Solution().insertionSortList(_f([6,5,3,1,8,7,2,4])))
print(Solution().insertionSortList(_f([6,5])))