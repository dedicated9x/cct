from typing import List
import torch

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


class TargetDecoder:

    def get_targets(
            self,
            canvas: MnistCanvas,
            anchors: List[MnistBox],
            iou_threshold: float = 0.5,
            nb_of_classes: int = 10,
    ) -> DigitDetectionModelTarget:
        raise NotImplementedError

    def get_predictions(
            self,
            model_output: DigitDetectionModelOutput,
    ) -> List[MnistBox]:
        raise

anchor_sizes = [
    (19, 19),
    (19, 15),
    (19, 13),
    (19, 11),
    (19, 5),
]

ds = ImagesDataset(split="train")
canvas = ds[0]
anchor_set = AnchorSet(anchor_sizes, k_grid=2)

decoder = TargetDecoder()
decoder.get_targets(canvas, anchor_set.list_mnistboxes)
