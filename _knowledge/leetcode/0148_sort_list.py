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

def _linkedlist_to_list(head):
    list_numbers = []
    ptr = head
    while ptr is not None:
        list_numbers.append(ptr)
        ptr = ptr.next
    return list_numbers

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

    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # TODO policz liczbe elementow
        if head is None:
            return None
        # n == 1
        if head.next is None:
            return head
        # n == 2
        elif head.next.next is None:
            first_elem = head
            second_elem = head.next
            if first_elem.val <= second_elem.val:
                return first_elem
            else:
                second_elem.next = first_elem
                first_elem.next = None
                return second_elem
        else:
            pass

        n_elements = self.get_n_elements(head)

        # Divide
        head_left, head_right = self.split_into_two_halves(head, n_elements)

        # Conquer
        sorted_left = self.sortList(head_left)
        sorted_right = self.sortList(head_right)

        # Combine
        _sorted = self.mergeTwoLists(sorted_left, sorted_right)
        return _sorted

    def get_n_elements(self, head):
        n_elements = 0
        ptr = head
        while ptr is not None:
            n_elements += 1
            ptr = ptr.next
        return n_elements

    def split_into_two_halves(self, head, n_elements):
        middle_element_idx = int(n_elements / 2)
        ptr = head
        for i in range(middle_element_idx):
            ptr = ptr.next

        head_right = ptr.next
        ptr.next = None
        return head, head_right



_f = list_to_linkedlist

# print(Solution().sortList(_f([-1])))
# print(Solution().sortList(_f([5, 3])))

print(Solution().sortList(_f([4,2,1,3])))
# print(Solution().sortList(_f([-1,5,3,4,0])))
