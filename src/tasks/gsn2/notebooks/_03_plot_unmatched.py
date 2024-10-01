import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('TkAgg')

from src.tasks.gsn2.anchor_set import AnchorSet
import numpy as np
import pandas as pd

from src.tasks.gsn2.dataset import ImagesDataset

# TODO random będzie subklasą
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

        self.anchor_set = anchor_set
        self.list_nonmatched = list_nonmatched

        return list_nonmatched



    # TODO to zostaje w notebooku
    def analyse_unmatched(self):
        # Define a function to calculate the anchor size as a tuple
        def _calculate_size(item):
            return (item.x_max - item.x_min, item.y_max - item.y_min)

        df = pd.DataFrame({'MnistBox': self.list_nonmatched})
        df['Size'] = df['MnistBox'].apply(_calculate_size)
        df = df[df['Size'].apply(lambda x: x[0] == 19)]
        df['Center'] = df['MnistBox'].apply(lambda x: x.center)

        xs = df['Center'].apply(lambda x: x[0]).tolist()
        ys = df['Center'].apply(lambda x: x[1]).tolist()

        fig, ax = plt.subplots(1, 1)
        ax.imshow(np.zeros((128, 128)))
        ax.scatter(ys, xs, s=3)

        self.anchor_set.grid.plot_on_ax(ax, color='red')
        # for box in self.list_nonmatched:
        #     box.plot_on_ax(ax)

        plt.show()

    # TODO stad tylko wez wspolrzedne
    def plot_next_unmatched(self, anchor_set: AnchorSet):
        for box in self.list_nonmatched:
            if box.size != (19, 19):
                continue
            anchor_set.display_all_anchors_in_loop(divisor=3, aux_box=box)


if __name__ == '__main__':
    anchor_sizes = [
        (19, 19),
        (19, 15),
        (19, 13),
        (19, 11),
        (19, 5),
    ]
    box_set = RandomMnistBoxSet(1000)
    anchor_set = AnchorSet(anchor_sizes, k_grid=2)
    box_set.match_with_anchorset(anchor_set, iou_threshold=0.5)

    box_set.analyse_unmatched()
    # box_set.plot_next_unmatched(anchor_set)
