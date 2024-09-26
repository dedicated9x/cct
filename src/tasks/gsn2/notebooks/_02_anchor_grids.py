import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('TkAgg')
import numpy as np
from src.tasks.gsn2.dataset import ImagesDataset


def get_anchor_grids():
    k_range = [2, 3, 4]
    anchor_grids = {k: [] for k in k_range}

    for k in k_range:
        m_max = int(128 / 2 ** k) - 1
        n_max = m_max
        for m in range(m_max + 1):
            for n in range(n_max + 1):
                new_x = m * 2 ** k + 2 ** (k-1)
                new_y = n * 2 ** k + 2 ** (k-1)
                anchor_grids[k].append((new_x, new_y))

    anchor_grids = {k: np.array(v) for k, v in anchor_grids.items()}
    return anchor_grids

if __name__ == '__main__':
    ds = ImagesDataset(split="train")
    great_number = 100
    for i in range(great_number):
        mnist_canvas = ds[i]

        fig, ax = plt.subplots()
        mnist_canvas.plot_on_ax(ax)

        anchor_grids = get_anchor_grids()
        for anchor_grid, color in zip(
            anchor_grids.values(),
            ["yellow", "orange", "red"]
        ):
            ax.scatter(anchor_grid[:, 0], anchor_grid[:, 1], color=color, s=3)

        plt.show()

        plt.waitforbuttonpress()  # Wait until a key or mouse click is pressed
        plt.close()  # Close the current figure after keypress/mouse click
