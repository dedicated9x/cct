from typing import  List, Tuple
from collections import deque

class Solution:
    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        m = len(board)
        n = len(board[0])

        # location = (3, 1)
        # nb = self.get_neighbours(location, m, n)

        points_on_border = self.get_points_on_border(m, n)
        d = deque()
        for point in points_on_border:
            x, y = point
            if board[x][y] == 'O':
                d.append(point)

        while d:
            next_point = d.popleft()
            x, y = next_point
            board[x][y] = 'B'
            for point in self.get_neighbours(next_point, m, n):
                x, y = point
                if board[x][y] == 'O':
                    d.append(point)

        self.replace(board, m, n, 'O', 'X')
        self.replace(board, m, n,'B', 'O')

    def replace(self, board, m, n, old_value, new_value) -> None:
        for row_idx in range(m):
            for col_idx in range(n):
                if board[row_idx][col_idx] == old_value:
                    board[row_idx][col_idx] = new_value

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

    def get_points_on_border(self, m, n):
        top = [(0, y) for y in range(n)]
        bottom = [(m - 1, y) for y in range(n)]
        left = [(x, 0) for x in range(1, m - 1)]
        right = [(x, n - 1) for x in range(1, m - 1)]

        all_points = top + bottom + left + right
        return all_points


print(Solution().solve(board=[
    ["X","X","X","X","X"],
    ["X","X","O","O","X"],
    ["X","X","X","X","X"],
    ["X","O","O","X","X"],
    ["X","O","X","X","X"]
]))