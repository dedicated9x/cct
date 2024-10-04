from typing import List

from src.tasks.gsn2.structures import DigitDetectionModelOutput, MnistBox
from src.tasks.gsn2.target_decoder import TargetDecoder


def _accuracy_single_canvas(
    predictions: List[MnistBox],
    gt_boxes: List[MnistBox],
) -> float:
    n_correct_predictions = 0
    for gt_box in gt_boxes:
        matched_predictions = []
        for prediction in predictions:
            if gt_box.iou_with(prediction) >= 0.5:
                matched_predictions.append(prediction)
        if len(matched_predictions) != 1:
            continue
        matched_prediction = matched_predictions[0]
        if matched_prediction.class_nb == gt_box.class_nb:
            n_correct_predictions += 1
    acc = n_correct_predictions / len(gt_boxes)
    return acc

def accuracy_single_canvas(
    model_output: DigitDetectionModelOutput,
    gt_boxes: List[MnistBox],
    iou_threshold: float,
    confidence_threshold: float
):
    predictions = TargetDecoder().get_predictions(model_output, iou_threshold, confidence_threshold)
    acc = _accuracy_single_canvas(predictions, gt_boxes)
    return acc