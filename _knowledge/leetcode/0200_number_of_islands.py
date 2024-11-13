from typing import  List, Tuple
from collections import deque

class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        """
        Do not return anything, modify board in-place instead.
        """
        m = len(grid)
        n = len(grid[0])

        island_counter = 0

        for idx_row in range(m):
            for idx_col in range(n):
                if grid[idx_row][idx_col] == '1':
                    island_counter += 1
                    island = self.get_island(grid, m, n, idx_row, idx_col)
                    for (x, y) in island:
                        grid[x][y] = '-1'

        return island_counter

    def get_island(self, grid, m, n, idx_row, idx_col) -> List[Tuple[int, int]]:
        d = deque()
        d.append((idx_row, idx_col))
        trawersed_points = []
        while d:
            point = d.popleft()
            trawersed_points.append(point)
            for neighbour in self.get_neighbours(point, m, n):
                x, y = neighbour
                if grid[x][y] == '1' and (neighbour not in trawersed_points) and (neighbour not in d):
                    d.append(neighbour)
        return trawersed_points

    def get_neighbours(self, xy: Tuple[int, int], m, n) -> List[Tuple[int, int]]:
        x, y = xy
        list_neighbours = [
            (x+1, y),
            (x-1, y),
            (x, y+1),
            (x, y-1)
        ]
        list_neighbours = [
            (x, y) for x, y in list_neighbours
            if 0 <= x <= m - 1 and 0 <= y <= n - 1
        ]
        return list_neighbours


print(Solution().numIslands(grid=[
    ["1","1","0","0","0"],
    ["1","1","0","0","0"],
    ["0","0","1","0","0"],
    ["0","0","0","1","1"]
]))

print(Solution().numIslands(grid=[["1"]]))
print(Solution().numIslands(grid=[["0"]]))