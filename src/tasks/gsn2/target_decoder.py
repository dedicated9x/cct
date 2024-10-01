from typing import List

import matplotlib.pyplot as plt
import torch
import numpy as np

from src.tasks.gsn2.dataset import MnistBox, MnistCanvas, ImagesDataset
from src.tasks.gsn2.anchor_set import AnchorSet


class DigitDetectionModelOutput:

    def __init__(
        self,
        anchors: List[MnistBox],
        classification_output: torch.Tensor,
        box_regression_output: torch.Tensor,
    ):
        self.anchors = anchors
        self.classification_output = classification_output
        self.box_regression_output = box_regression_output


class DigitDetectionModelTarget:

    def __init__(
            self,
            classification_target: torch.Tensor,
            box_regression_target: torch.Tensor,
            matched_anchors: List[int],
    ):
        self.classification_target = classification_target
        self.box_regression_target = box_regression_target
        self.matched_anchors = matched_anchors


# TODO leciec z tym z powrotem. Może rozbij na dwa.
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


class TargetDecoder:

    def get_targets(
            self,
            canvas: MnistCanvas,
            anchors: List[MnistBox],
            iou_threshold: float = 0.5,
            nb_of_classes: int = 10,
    ) -> DigitDetectionModelTarget:
        classification_target = torch.zeros((len(anchors), 10))
        box_regression_target = torch.zeros((len(anchors), 4))
        matched_anchors = []

        for idx_anchor, anchor in enumerate(anchors):
            list_ious_anchor = []
            for box in canvas.boxes:
                iou = box.iou_with(anchor)
                list_ious_anchor.append(iou)
            if max(list_ious_anchor) >= iou_threshold:
                idx_gt_best = np.array(list_ious_anchor).argmax()
                gt_best = canvas.boxes[idx_gt_best]

                box_regression_target[idx_anchor, 0] = gt_best.x_min - anchor.x_min
                box_regression_target[idx_anchor, 1] = gt_best.x_max - anchor.x_max
                box_regression_target[idx_anchor, 2] = gt_best.y_min - anchor.y_min
                box_regression_target[idx_anchor, 3] = gt_best.y_max - anchor.y_max

                classification_target[idx_anchor, gt_best.class_nb] = 1
                matched_anchors.append(idx_anchor)

            else:
                continue

        return DigitDetectionModelTarget(
            classification_target,
            box_regression_target,
            matched_anchors
        )

    def get_predictions(
            self,
            model_output: DigitDetectionModelOutput,
    ) -> List[MnistBox]:
        raise

if __name__ == '__main__':
    anchor_sizes = [
        (19, 19),
        (19, 15),
        (19, 13),
        (19, 11),
        (19, 5),
    ]
    ds = ImagesDataset(split="train")
    anchor_set = AnchorSet(anchor_sizes, k_grid=2)
    decoder = TargetDecoder()

    for i in range(100):
        canvas = ds[i]
        target = decoder.get_targets(canvas, anchor_set.list_mnistboxes)

        matched_anchors = [
            elem for idx, elem in enumerate(anchor_set.list_mnistboxes)
            if idx in target.matched_anchors
        ]

        fig, ax = plt.subplots()
        canvas.plot_on_ax(ax)
        for anchor in matched_anchors:
            anchor.plot_on_ax(ax, color="white")

        plt.waitforbuttonpress()  # Wait until a key or mouse click is pressed
        plt.close()  # Close the current figure after keypress/mouse click


