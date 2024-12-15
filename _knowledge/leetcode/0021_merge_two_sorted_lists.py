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

class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        zero_node = ListNode(0, None)
        ptr1 = list1
        ptr2 = list2

        if ptr1 is None:
            return ptr2
        if ptr2 is None:
            return ptr1

        current_node = zero_node
        while (ptr1 is not  None) and (ptr2 is not None):
            if ptr1.val <= ptr2.val:
                current_node.next = ptr1
                ptr1 = ptr1.next
                current_node = current_node.next
            else:
                current_node.next = ptr2
                ptr2 = ptr2.next
                current_node = current_node.next

        if ptr1 is None:
            non_empty_ptr = ptr2
        else:
            non_empty_ptr = ptr1

        current_node.next = non_empty_ptr
        return zero_node.next


_f = list_to_linkedlist

# print(Solution().mergeTwoLists(_f([1, 2, 4]), _f([1, 3, 4])))
print(Solution().mergeTwoLists(_f([2]), _f([1])))
