from typing import List

class TreeNode:
    def __init__(self, val=0):
        self.val = val
        self.edges_in = []
        self.edges_out = []
        self.type = None
        self.state = None

    def __repr__(self):
        return f"{self.val} in:{self.edges_in} out{self.edges_out} type:{self.type} state: {self.state}"

    def _calculate_type(self):
        len_in = len(self.edges_in)
        len_out = len(self.edges_out)

        if len_in >= 1 and len_out >= 1:
            self.type = "bidirectional"
        elif len_in == 0 and len_out >= 1:
            self.type = "all_out"
        elif len_in >= 1 and len_out == 0:
            self.type = "all_in"
        else:
            self.type = "isolated"

    def _calculate_state(self):
        if self.state == "traversed":
            self.state = "traversed"
        else:
            if self.type in ["isolated", "all_out"]:
                self.state = "accessible"
            else:
                self.state = "inaccessible"

    def calculate_params(self):
        self._calculate_type()
        self._calculate_state()

class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        list_nodes = [TreeNode(idx) for idx in range(numCourses)]

        for edge in prerequisites:
            edge_out, edge_in = edge
            list_nodes[edge_out].edges_in.append(edge_in)
            list_nodes[edge_in].edges_out.append(edge_out)

        for node in list_nodes:
            node.calculate_params()

        list_order = []

        while "accessible" in [node.state for node in list_nodes]:
            for node in list_nodes:
                if node.type == "isolated" and node.state == "accessible":
                    list_order.append(node.val)
                    node.state = "traversed"
                elif node.type == "all_out" and node.state == "accessible":

                    for child_idx in node.edges_out:
                        child = list_nodes[child_idx]
                        child.edges_in.remove(node.val)
                        child.calculate_params()

                    node.edges_out = []
                    node.calculate_params()

                    list_order.append(node.val)
                    node.state = "traversed"
                else:
                    pass
            a = 2

        if "inaccessible" in [node.state for node in list_nodes]:
            return []
        else:
            return list_order


# print(Solution().findOrder(numCourses = 4, prerequisites = [[1,0],[2,0],[3,1],[3,2]]))

# print(Solution().findOrder(numCourses = 11, prerequisites = [
#     [2,1],
#     [3,2],
#     [4,2],
#     [5,3],
#     [5,4],
#     [7,6],
#     [7,8],
#     [9,8],
#     [10,9],
#     [8,10],
#     [9,2],
# ]))

print(Solution().findOrder(numCourses = 2, prerequisites = [[0,1]]))
