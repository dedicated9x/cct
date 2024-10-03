import torch

from src.tasks.gsn2.anchor_set import AnchorSet
from src.tasks.gsn2.target_decoder import TargetDecoder
from src.tasks.gsn2.arch import DigitDetectionModelOutput

from src.tasks.gsn2.dataset import ImagesDataset



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

    classification_output = torch.load("/home/admin2/Documents/repos/cct/.EXCLUDED/outputs/clf_output.pt")
    box_regression_output = torch.load("/home/admin2/Documents/repos/cct/.EXCLUDED/outputs/boxreg_output.pt")

    model_output = DigitDetectionModelOutput(
        anchor_set.list_mnistboxes,
        classification_output,
        box_regression_output
    )


