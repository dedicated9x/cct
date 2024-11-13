# Definition for a Node.
class Node:
    def __init__(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []



from typing import Optional, List
from collections import deque

def edges_to_graph(edges: List):
    idx2node = {}
    for idx, _ in enumerate(edges):
        idx2node[idx + 1] = Node(val=idx + 1)

    for idx, neighbors_idxs in enumerate(edges):
        neighbors = [idx2node[idx] for idx in neighbors_idxs]
        idx2node[idx + 1].neighbors = neighbors

    return idx2node[1]

class Solution:
    def cloneGraph(self, node: Optional['Node']) -> Optional['Node']:
        if node is None:
            return node

        d = deque()
        traversed_nodes = []

        d.append(node)
        while d:
            next_node = d.popleft()
            traversed_nodes.append(next_node)
            for node_ in next_node.neighbors:
                if node_ not in traversed_nodes and node_ not in d:
                    d.append(node_)

        traversed_nodes = sorted(traversed_nodes, key=lambda x: x.val)
        edges = [[f.val for f in e.neighbors] for e in traversed_nodes]
        graph = edges_to_graph(edges)
        return graph




edges = [[2,4],[1,3],[2,4],[1,3]]
graph = edges_to_graph(edges)
print(Solution().cloneGraph(graph))