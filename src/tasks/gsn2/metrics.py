from typing import List, Dict, Optional, Union

import torch

from src.tasks.gsn2.structures import DigitDetectionModelOutput, MnistBox
from src.tasks.gsn2.target_decoder import TargetDecoder


def _get_metrics_sample(
    predictions: List[MnistBox],
    gt_boxes: List[MnistBox],
) -> Dict[str, float]:
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

    if len(predictions) > 0:
        precision = n_correct_predictions / len(predictions)
    else:
        precision = 0

    if n_correct_predictions == len(gt_boxes) == len(predictions):
        accuracy = 1
    else:
        accuracy = 0

    return {
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy
    }

def get_metrics_sample(
    model_output: DigitDetectionModelOutput,
    gt_boxes: List[MnistBox],
    iou_threshold: float,
    confidence_threshold: float
) -> Dict[str, float]:
    predictions = TargetDecoder().get_predictions(model_output, iou_threshold, confidence_threshold)
    metrics = _get_metrics_sample(predictions, gt_boxes)
    return metrics

def list_average(
        _list: List[float],
        weights: Optional[List[Union[float, int]]]
):
    sum_weights= sum(weights)
    weights = [e / sum_weights for e in weights]
    avg = sum([w * x for w, x in zip(weights, _list)])
    return avg

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
        )['recall']
        list_recalls.append(recall)
        list_gt_box_counts.append(len(gt_boxes_sample))

    weighted_recall = list_average(list_recalls, list_gt_box_counts)
    return weighted_recall
