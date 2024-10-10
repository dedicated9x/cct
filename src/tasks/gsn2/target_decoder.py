from typing import List

import torch
import torch.nn.functional as F
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('TkAgg')

from src.tasks.gsn2.structures import MnistBox, MnistCanvas, DigitDetectionModelOutput

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

    def as_dict_of_tensors(self):
        matched_anchors = torch.tensor(self.matched_anchors)

        matched_anchors_padded = torch.nn.functional.pad(
            matched_anchors,
            pad= (0, len(self.classification_target) - matched_anchors.shape[0]),
            value=-1
        )
        return {
            "classification_target": self.classification_target,
            "box_regression_target": self.box_regression_target,
            "matched_anchors": matched_anchors_padded
        }

def get_nms_filter(model_output: DigitDetectionModelOutput, iou_threshold):
    xs_min = torch.Tensor([e.x_min for e in model_output.anchors]) + model_output.box_regression_output[:, 0]
    ys_min = torch.Tensor([e.y_min for e in model_output.anchors]) + model_output.box_regression_output[:, 2]
    xs_max = torch.Tensor([e.x_max for e in model_output.anchors]) + model_output.box_regression_output[:, 1]
    ys_max = torch.Tensor([e.y_max for e in model_output.anchors]) + model_output.box_regression_output[:, 3]

    boxes = torch.stack([xs_min, ys_min, xs_max, ys_max])
    boxes = boxes.transpose(0, 1)

    scores = torch.sigmoid(model_output.classification_output).max(dim=1).values
    nms_filter = torchvision.ops.nms(boxes, scores, iou_threshold)
    return nms_filter

def get_confidence_filter(model_output: DigitDetectionModelOutput, confidence_threshold):
    confidence_filter = torch.sigmoid(model_output.classification_output).max(dim=1).values > confidence_threshold

    # Convert from boolean mask to tensor of indices
    confidence_filter = torch.nonzero(confidence_filter, as_tuple=False).squeeze()
    return confidence_filter

def logical_and_filter(filter1, filter2):
    # TODO one niemusza byc tensorami
    def tensor_to_set(tensor):
        if tensor.dim() == 0:
            return {tensor.item()}  # Zamień skalara na zbiór z jednym elementem
        else:
            return set(tensor.tolist())  # Zamień tensor wielowymiarowy na zbiór elementów


    # Convert tensors to sets and find intersection
    filter1_set = tensor_to_set(filter1)
    filter2_set = tensor_to_set(filter2)

    # Find the intersection
    intersection = filter1_set & filter2_set

    # Convert the result back to a tensor
    return torch.tensor(list(intersection))

def _get_predictions(
        model_output: DigitDetectionModelOutput,
        idxs_predictions: List[int],
        anchors: List[MnistBox]
) -> List[MnistBox]:

    list_predictions = []
    for idx in idxs_predictions:
        class_nb = model_output.classification_output[idx].argmax().item()

        x_min = int(anchors[idx].x_min + model_output.box_regression_output[idx][0])
        x_max = int(anchors[idx].x_max + model_output.box_regression_output[idx][1])
        y_min = int(anchors[idx].y_min + model_output.box_regression_output[idx][2])
        y_max = int(anchors[idx].y_max + model_output.box_regression_output[idx][3])

        prediction = MnistBox(x_min, y_min, x_max, y_max, class_nb)
        list_predictions.append(prediction)

    return list_predictions

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
            iou_threshold: float,
            confidence_threshold: float
    ) -> List[MnistBox]:
        nms_filter = get_nms_filter(model_output, iou_threshold)
        confidence_filter = get_confidence_filter(model_output, confidence_threshold)
        idxs_predictions_ = logical_and_filter(nms_filter, confidence_filter)
        predictions = _get_predictions(model_output, idxs_predictions_.tolist(), model_output.anchors)
        return predictions

if __name__ == '__main__':
    from src.tasks.gsn2.dataset import ImagesDataset
    from src.tasks.gsn2.anchor_set import AnchorSet

    anchor_sizes = [
        (19, 19),
        (19, 15),
        (19, 13),
        (19, 11),
        (19, 5),
    ]
    ds = ImagesDataset(split="train", size=1024)
    anchor_set = AnchorSet(anchor_sizes, k_grid=2)
    decoder = TargetDecoder()

    for i in range(100):
        canvas = ds.get_canvas(i)
        target = decoder.get_targets(
            canvas,
            anchor_set.list_mnistboxes,
            # iou_threshold=0.3
        )

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

