import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('TkAgg')

from typing import List, Tuple
from src.tasks.gsn2.notebooks._03_plot_unmatched_part1 import Grid
from src.tasks.gsn2.dataset import MnistBox
import numpy as np
import torch
import pandas as pd

class AnchorSet:
    def __init__(self, anchor_sizes: List[Tuple], k_grid: int):
        grid = Grid(k=3)

        n_grid_centers = grid.grid_flattened.shape[0]
        n_anchor_sizes = len(anchor_sizes)

        # Placeholder below
        set_as_tensor3d = np.full((n_grid_centers, n_anchor_sizes, 4), None)
        for idx_grid_center in range(n_grid_centers):
            for idx_anchor_size in range(n_anchor_sizes):
                set_as_tensor3d[idx_grid_center, idx_anchor_size, 0:2] = grid.grid_flattened[idx_grid_center]
                set_as_tensor3d[idx_grid_center, idx_anchor_size, 2:4] = anchor_sizes[idx_anchor_size]

        assert not np.any(set_as_tensor3d == None)
        set_as_tensor3d = torch.Tensor(np.array(set_as_tensor3d, dtype=int))

        # 2nd flattening here
        set_as_tensor2d = set_as_tensor3d.flatten(start_dim=0, end_dim=1)

        list_mnistboxes = []
        for row in set_as_tensor2d:
            mnistbox = MnistBox.from_size_and_center(
                center_x=int(row[0].item()),
                center_y=int(row[1].item()),
                size_x=int(row[2].item()),
                size_y=int(row[3].item()),
            )
            list_mnistboxes.append(mnistbox)

        self.anchor_sizes = anchor_sizes
        self.list_mnistboxes = list_mnistboxes

    def present_anchors(self) -> None:
        # Define a function to calculate the anchor size as a tuple
        def _calculate_anchor_size(item):
            return (item.x_max - item.x_min, item.y_max - item.y_min)

        df =  pd.DataFrame({'MnistBox': self.list_mnistboxes})
        df['Anchor Size'] = df['MnistBox'].apply(_calculate_anchor_size)
        grouped_dict = df.groupby('Anchor Size')['MnistBox'].apply(list).to_dict()

        fig, axes = plt.subplots(2, (len(self.anchor_sizes) + 1) // 2, figsize=(10, 5))
        for anchor_size, ax in zip(
            grouped_dict.keys(),
            axes.flatten()[:len(grouped_dict)]
        ):
            list_mnistboxes = grouped_dict[anchor_size]
            bg = np.zeros((128, 128))
            ax.imshow(bg)
            for box in list_mnistboxes:
                box.plot_on_ax(
                    ax,
                    plot_text=False
                )
            ax.set_xlabel(str(anchor_size))


if __name__ == '__main__':
    anchor_sizes = [
        (19, 19),
        (19, 15),
        (19, 13),
        (19, 11),
        (19, 5),
    ]
    anchor_set = AnchorSet(anchor_sizes, k_grid=3)
    anchor_set.present_anchors()