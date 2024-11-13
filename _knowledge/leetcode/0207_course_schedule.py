from typing import List

class Node:
    def __init__(self, val = 0):
        self.val = val
        self.edges_in = []
        self.edges_out = []
        self.type = None
        self.is_removable = None

    def __str__(self):
        return f"{self.val} in:{self.edges_in} out:{self.edges_out} type:{self.type}"

    def calculate_flags(self):
        self._calculate_type()
        self._evaluate_is_removable()

    def _calculate_type(self):
        len_in = len(self.edges_in)
        len_out = len(self.edges_out)

        if len_in > 0 and len_out > 0:
            self.type = "bidirectional"
        elif len_in > 0 and len_out == 0:
            self.type = "all_in"
        elif len_in == 0 and len_out > 0:
            self.type = "all_out"
        else:
            self.type = "isolated"

    def _evaluate_is_removable(self):
        if self.type in ["all_in", "all_out"]:
            self.is_removable = True
        else:
            self.is_removable = False

class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        if numCourses == 1:
            return True

        list_nodes = [Node(val=idx) for idx in range(numCourses)]
        for edge in prerequisites:
            node_out, node_in = edge

            # It's better to have hashable elements.
            edge = (node_out, node_in)

            list_nodes[node_out].edges_out.append(edge)
            list_nodes[node_in].edges_in.append(edge)

        for node in list_nodes:
            node.calculate_flags()

        while any([e.is_removable for e in list_nodes]):
            for node in list_nodes:
                if node.type == "all_out":
                    a = 2
        # str(list_nodes[0])


print(Solution().canFinish(numCourses = 11, prerequisites = [
    [0, 7],
    [1, 3],
    [2, 3],
    [3, 4],
    [4, 5],
    [5, 6],
    [6, 4],
    [5, 7],
    [6, 8],
    [6, 10],
    [8, 9],
    [10, 9],
]))
# print(Solution().canFinish(numCourses = 2, prerequisites = [[1,0]]))
# print(Solution().canFinish(numCourses = 2, prerequisites = [[1,0],[0,1]]))
# print(Solution().canFinish(numCourses = 6, prerequisites = [[0,1], [1,2], [2, 0], [4, 3], [5,3]]))