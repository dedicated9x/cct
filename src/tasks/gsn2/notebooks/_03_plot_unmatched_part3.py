import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('TkAgg')

from typing import List, Tuple
from src.tasks.gsn2.notebooks._03_plot_unmatched_part2 import AnchorSet
from src.tasks.gsn2.dataset import MnistBox
import numpy as np
import torch
import pandas as pd

from src.tasks.gsn2.dataset import ImagesDataset

class RandomMnistBoxSet:
    def __init__(self, n_boxes:int):
        ds = ImagesDataset(split="train")

        list_boxes = []
        i = 0
        while len(list_boxes) < n_boxes:
            mnist_canvas = ds[i]
            list_boxes += mnist_canvas.boxes
            i += 1
        list_boxes = list_boxes[:n_boxes]

        self.list_boxes = list_boxes

    def match_with_anchorset(self, anchor_set: AnchorSet, iou_threshold: float):
        n_boxes = len(self.list_boxes)
        n_anchors = len(anchor_set.list_mnistboxes)

        grid_ious = np.full((n_boxes, n_anchors), None)
        for box_idx in range(n_boxes):
            for anchor_idx in range(n_anchors):
                box = self.list_boxes[box_idx]
                anchor = anchor_set.list_mnistboxes[anchor_idx]
                iou = box.iou_with(anchor)
                grid_ious[box_idx, anchor_idx] = iou

        assert not np.any(grid_ious == None)

        filter_is_above_threshold = (grid_ious.max(axis=1) > iou_threshold).tolist()

        list_nonmatched = [
            box for box, is_above_threshold in zip(self.list_boxes, filter_is_above_threshold)
            if not is_above_threshold
        ]

        self.list_nonmatched = list_nonmatched

    def analyse_unmatched(self):
        # Define a function to calculate the anchor size as a tuple
        def _calculate_size(item):
            return (item.x_max - item.x_min, item.y_max - item.y_min)

        df = pd.DataFrame({'MnistBox': self.list_nonmatched})
        df['Size'] = df['MnistBox'].apply(_calculate_size)

        fig, ax = plt.subplots(1, 1)
        ax.imshow(np.zeros((128, 128)))
        for box in self.list_nonmatched:
            box.plot_on_ax(ax)


    def plot_next_unmatched(self):
        # TODO trzeba to zrobic, aby miec pewnosc, ze nie dzieje sie tu jakichs totalny syf
        raise NotImplementedError


if __name__ == '__main__':
    anchor_sizes = [
        (19, 19),
        (19, 15),
        (19, 13),
        (19, 11),
        (19, 5),
    ]
    box_set = RandomMnistBoxSet(1000)
    anchor_set = AnchorSet(anchor_sizes, k_grid=3)
    box_set.match_with_anchorset(anchor_set, iou_threshold=0.5)
    box_set.analyse_unmatched()
