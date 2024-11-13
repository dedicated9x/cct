from typing import List
from collections import deque

class Node:
    def __init__(self, val = 0, targets = None):
        self.val = val
        self.targets = targets if targets is not None else []

class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        if numCourses == 1:
            return True

        list_nodes = [Node(val=idx) for idx in range(numCourses)]
        for idx_node, idx_target in prerequisites:
            list_nodes[idx_node].targets.append(list_nodes[idx_target])

        for idx_start, _ in enumerate(list_nodes):
            if self.has_loop(idx_start, list_nodes):
                return False

        # Doesn't have any loops.
        return True

    def has_loop(self, idx_start, list_nodes) -> bool:
        start_node = list_nodes[idx_start]

        d = deque()
        for node in start_node.targets:
            d.append(node)
        trawersed_nodes = []
        while d:
            next_node = d.popleft()
            trawersed_nodes.append(next_node)
            for node in next_node.targets:
                if (node not in trawersed_nodes) and (node not in d):
                    d.append(node)

        if start_node in trawersed_nodes:
            return True
        else:
            return False

print(Solution().canFinish(numCourses = 2, prerequisites = [[1,0]]))
print(Solution().canFinish(numCourses = 2, prerequisites = [[1,0],[0,1]]))
print(Solution().canFinish(numCourses = 6, prerequisites = [[0,1], [1,2], [2, 0], [4, 3], [5,3]]))