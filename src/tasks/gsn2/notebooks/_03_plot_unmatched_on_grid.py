from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('TkAgg')

from src.tasks.gsn2.grid import Grid
from src.tasks.gsn2.anchor_set import AnchorSet
from src.tasks.gsn2.dataset import ImagesDataset
from src.tasks.gsn2.structures import MnistBox

def get_random_mnistboxes(n_boxes:int):
    ds = ImagesDataset(split="train", size=10000)

    list_boxes = []
    i = 0
    while len(list_boxes) < n_boxes:
        mnist_canvas = ds[i]
        list_boxes += mnist_canvas.boxes
        i += 1
    list_boxes = list_boxes[:n_boxes]

    return list_boxes

def match_with_anchorset(
        list_boxes: List[MnistBox],
        list_anchors: List[MnistBox],
        iou_threshold: float
):
    n_boxes = len(list_boxes)
    n_anchors = len(list_anchors)

    grid_ious = np.full((n_boxes, n_anchors), None)
    for box_idx in range(n_boxes):
        for anchor_idx in range(n_anchors):
            box = list_boxes[box_idx]
            anchor = list_anchors[anchor_idx]
            iou = box.iou_with(anchor)
            grid_ious[box_idx, anchor_idx] = iou

    assert not np.any(grid_ious == None)

    filter_is_above_threshold = (grid_ious.max(axis=1) > iou_threshold).tolist()

    list_nonmatched = [
        box for box, is_above_threshold in zip(list_boxes, filter_is_above_threshold)
        if not is_above_threshold
    ]

    return list_nonmatched


def plot_unmatched_on_grid(list_nonmatched: List[MnistBox], grid: Grid):
    # Define a function to calculate the anchor size as a tuple
    def _calculate_size(item):
        return (item.x_max - item.x_min, item.y_max - item.y_min)

    df = pd.DataFrame({'MnistBox': list_nonmatched})
    df['Size'] = df['MnistBox'].apply(_calculate_size)
    df = df[df['Size'].apply(lambda x: x[0] == 19)]
    df['Center'] = df['MnistBox'].apply(lambda x: x.center)

    xs = df['Center'].apply(lambda x: x[0]).tolist()
    ys = df['Center'].apply(lambda x: x[1]).tolist()

    fig, ax = plt.subplots(1, 1)
    ax.imshow(np.zeros((128, 128)))
    ax.scatter(ys, xs, s=3)

    grid.plot_on_ax(ax, color='red')
    plt.show()

if __name__ == '__main__':
    anchor_sizes = [
        (19, 19),
        (19, 15),
        (19, 13),
        (19, 11),
        (19, 5),
    ]
    list_boxes = get_random_mnistboxes(n_boxes=1000)
    anchor_set = AnchorSet(anchor_sizes, k_grid=3)
    list_nonmatched = match_with_anchorset(list_boxes, anchor_set.list_mnistboxes, iou_threshold=0.5)

    plot_unmatched_on_grid(list_nonmatched, anchor_set.grid)
