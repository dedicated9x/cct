import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('TkAgg')
import numpy as np
import torch


class Grid:
    def __init__(self, k:int):
        size = int(128 / 2 ** k)

        # Placeholder below
        grid_as_tensor = np.full((size, size, 2), None)

        for m in range(size):
            for n in range(size):
                new_x = m * 2 ** k + 2 ** (k-1)
                new_y = n * 2 ** k + 2 ** (k-1)
                grid_as_tensor[m, n, 0] = new_x
                grid_as_tensor[m, n, 1] = new_y

        assert not np.any(grid_as_tensor == None)
        grid_as_tensor = torch.Tensor(np.array(grid_as_tensor, dtype=int))

        # 1st flattening here.
        grid_flattened = grid_as_tensor.flatten(start_dim=0, end_dim=1)

        self.size = size
        self.grid_as_tensor = grid_as_tensor
        self.grid_flattened = grid_flattened


    def plot_on_ax(self, ax, color="r"):
        ax.scatter(
            self.grid_flattened[:, 0],
            self.grid_flattened[:, 1],
            color=color,
            s=3
        )


if __name__ == '__main__':
    bg = np.zeros((128, 128))

    fig, ax = plt.subplots()
    ax.imshow(bg)

    for k, color in zip(
        [2, 3, 4],
        ["yellow", "orange", "red"]

    ):
        grid = Grid(k)
        grid.plot_on_ax(ax, color=color)
