
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('TkAgg')

from src.tasks.gsn2.anchor_set import AnchorSet
from src.tasks.gsn2.dataset import ImagesDataset
from src.tasks.gsn2.target_decoder import TargetDecoder
from src.tasks.gsn2.structures import DigitDetectionModelOutput
from src.tasks.gsn2.metrics import get_metrics_sample

def plot_predictions(model_output, canvas_image, limit):
    fig, axes = plt.subplots(3, 3)

    idx_ax = 0
    for confidence_threshold in [0.5, 0.6, 0.7]:
        for iou_threshold in [0.3, 0.5, 0.7]:
            ax = axes.flatten()[idx_ax]
            idx_ax += 1

            predictions = TargetDecoder().get_predictions(model_output, iou_threshold, confidence_threshold)

            # Limit number of plotted predictions
            if len(predictions) >= limit:
                predictions = np.random.choice(predictions, limit, replace=False).tolist()

            ax.imshow(canvas_image)
            for box in predictions:
                box.plot_on_ax(ax)

            ax.set_xlabel(f"iou={iou_threshold}, conf={confidence_threshold}")

    # Leave possibility to log the figure
    return fig



if __name__ == '__main__':
    anchor_sizes = [
        (19, 19),
        (19, 15),
        (19, 13),
        (19, 11),
        (19, 5),
    ]
    anchor_set_ = AnchorSet(anchor_sizes, k_grid=2)
    ds_val_ = ImagesDataset(
        "val",None, 0.5, anchors=anchor_set_.list_mnistboxes
    )
    canvas_ = ds_val_.get_canvas(4)

    classification_output_ = torch.load("/home/admin2/Documents/repos/cct/.EXCLUDED/outputs/clf_output.pt")
    box_regression_output_ = torch.load("/home/admin2/Documents/repos/cct/.EXCLUDED/outputs/boxreg_output.pt")

    model_output_ = DigitDetectionModelOutput(
        anchor_set_.list_mnistboxes,
        classification_output_,
        box_regression_output_
    )

    plot_predictions(model_output_, canvas_.image, limit=100)
    plt.show()

    acc = get_metrics_sample(
        model_output=model_output_,
        gt_boxes=canvas_.boxes,
        iou_threshold=0.5,
        confidence_threshold=0.7
    )

    print(acc)





