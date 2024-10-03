from typing import List

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('TkAgg')

from src.tasks.gsn2.anchor_set import AnchorSet
from src.tasks.gsn2.dataset import ImagesDataset
from src.tasks.gsn2.target_decoder import TargetDecoder
from src.tasks.gsn2.structures import DigitDetectionModelOutput


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

    classification_output_ = torch.load("/home/admin2/Documents/repos/cct/.EXCLUDED/outputs/clf_output.pt")
    box_regression_output_ = torch.load("/home/admin2/Documents/repos/cct/.EXCLUDED/outputs/boxreg_output.pt")

    model_output_ = DigitDetectionModelOutput(
        anchor_set.list_mnistboxes,
        classification_output_,
        box_regression_output_
    )

    fig, axes = plt.subplots(3, 3)


    # confidence_threshold_ = 0.6
    # iou_threshold_ = 0.5
    limit = 100

    idx_ax = 0
    for confidence_threshold_ in [0.5, 0.6, 0.7]:
        for iou_threshold_ in [0.3, 0.5, 0.7]:

            ax = axes.flatten()[idx_ax]
            idx_ax += 1

            predictions = TargetDecoder().get_predictions(model_output_, iou_threshold_, confidence_threshold_)

            # Limit number of plotted predictions
            if len(predictions) >= limit:
                predictions = np.random.choice(predictions, limit, replace=False).tolist()

            canvas.plot_on_ax(ax, boxes=predictions)
            ax.set_xlabel(f"iou={iou_threshold_}, conf={confidence_threshold_}")

    plt.show()


