from typing import List

import torch

from src.tasks.gsn2.structures import DigitDetectionModelOutput, MnistBox
from src.tasks.gsn2.target_decoder import TargetDecoder


def _get_metrics_sample(
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
    recall = n_correct_predictions / len(gt_boxes)
    return recall

def get_metrics_sample(
    model_output: DigitDetectionModelOutput,
    gt_boxes: List[MnistBox],
    iou_threshold: float,
    confidence_threshold: float
):
    predictions = TargetDecoder().get_predictions(model_output, iou_threshold, confidence_threshold)
    recall = _get_metrics_sample(predictions, gt_boxes)
    return recall

def get_metrics_batch(
    anchors: List[MnistBox],
    classification_output: torch.Tensor,
    box_regression_output: torch.Tensor,
    gt_boxes: List[List[MnistBox]],
    iou_threshold: float,
    confidence_threshold: float
):
    list_recalls = []
    list_gt_box_counts = []
    for idx_sample, _ in enumerate(classification_output):
        model_output_sample = DigitDetectionModelOutput(
            anchors,
            classification_output[idx_sample],
            box_regression_output[idx_sample],
        )
        gt_boxes_sample = gt_boxes[idx_sample]

        recall = get_metrics_sample(
            model_output=model_output_sample,
            gt_boxes=gt_boxes_sample,
            iou_threshold=iou_threshold,
            confidence_threshold=confidence_threshold
        )
        list_recalls.append(recall)
        list_gt_box_counts.append(len(gt_boxes_sample))

    acc = sum([x * y for x, y in zip(list_recalls, list_gt_box_counts)]) / sum(list_gt_box_counts)
    return acc
