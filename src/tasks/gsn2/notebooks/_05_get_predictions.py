from typing import List

import torch
import torchvision
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('TkAgg')

from src.tasks.gsn2.anchor_set import AnchorSet
from src.tasks.gsn2.target_decoder import TargetDecoder
from src.tasks.gsn2.arch import DigitDetectionModelOutput
from src.tasks.gsn2.structures import MnistBox

from src.tasks.gsn2.dataset import ImagesDataset

def get_nms_filter(model_output: DigitDetectionModelOutput):
    xs_min = torch.Tensor([e.x_min for e in model_output.anchors]) + model_output.box_regression_output[:, 0]
    ys_min = torch.Tensor([e.y_min for e in model_output.anchors]) + model_output.box_regression_output[:, 2]
    xs_max = torch.Tensor([e.x_max for e in model_output.anchors]) + model_output.box_regression_output[:, 1]
    ys_max = torch.Tensor([e.y_max for e in model_output.anchors]) + model_output.box_regression_output[:, 3]

    boxes = torch.stack([xs_min, ys_min, xs_max, ys_max])
    boxes = boxes.transpose(0, 1)

    scores = torch.sigmoid(model_output.classification_output).max(dim=1).values
    nms_filter = torchvision.ops.nms(boxes, scores, iou_threshold=0.5)
    return nms_filter

def get_confidence_filter(model_output: DigitDetectionModelOutput, confidence_threshold):
    confidence_filter = torch.sigmoid(model_output.classification_output).max(dim=1).values > confidence_threshold

    # Convert from boolean mask to tensor of indices
    confidence_filter = torch.nonzero(confidence_filter, as_tuple=False).squeeze()
    return confidence_filter

def logical_and_filter(filter1, filter2):
    # TODO one niemusza byc tensorami
    # Convert tensors to sets and find intersection
    filter1_set = set(filter1.tolist())
    filter2_set = set(filter2.tolist())

    # Find the intersection
    intersection = filter1_set & filter2_set

    # Convert the result back to a tensor
    return torch.tensor(list(intersection))

def get_predictions(
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



if __name__ == '__main__':
    anchor_sizes = [
        (19, 19),
        (19, 15),
        (19, 13),
        (19, 11),
        (19, 5),
    ]
    anchor_set = AnchorSet(anchor_sizes, k_grid=2)
    ds_val = ImagesDataset(
        "val",None, 0.5, anchors=anchor_set.list_mnistboxes
    )
    canvas = ds_val.get_canvas(0)


    confidence_threshold_ = 0.6


    classification_output_ = torch.load("/home/admin2/Documents/repos/cct/.EXCLUDED/outputs/clf_output.pt")
    box_regression_output_ = torch.load("/home/admin2/Documents/repos/cct/.EXCLUDED/outputs/boxreg_output.pt")

    model_output_ = DigitDetectionModelOutput(
        anchor_set.list_mnistboxes,
        classification_output_,
        box_regression_output_
    )

    nms_filter = get_nms_filter(model_output_)
    confidence_filter = get_confidence_filter(model_output_, confidence_threshold_)
    idxs_predictions_ = logical_and_filter(nms_filter, confidence_filter)
    print(len(idxs_predictions_))
    predictions = get_predictions(model_output_, idxs_predictions_.tolist(), anchor_set.list_mnistboxes)

    fig, ax = plt.subplots()
    canvas.plot_on_ax(ax, boxes=predictions)

    plt.show()


