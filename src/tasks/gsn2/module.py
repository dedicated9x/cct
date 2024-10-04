import torch
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import matplotlib;matplotlib.use('TkAgg')
import numpy as np
import wandb

from src.common.module import BaseModule
from src.tasks.gsn2.anchor_set import AnchorSet
from src.tasks.gsn2.arch import MyNet32, DigitDetectionModelOutput
from src.tasks.gsn2.dataset import ImagesDataset
from src.tasks.gsn2.target_decoder import TargetDecoder
from src.tasks.gsn2.notebooks._05_get_predictions import plot_predictions
from src.tasks.gsn2.metrics import get_metrics_batch



def custom_collate_fn(batch):
    items_stackable = [
        "classification_target", "box_regression_target",
        "matched_anchors", "canvas"
    ]
    items_nonstackable = [
        "boxes"
    ]
    items_all = items_stackable + items_nonstackable

    output_batch = {}
    batch_size = len(batch)
    for item_name in items_all:
        batch_item = [batch[idx][item_name] for idx in range(batch_size)]
        if item_name in items_stackable:
            batch_item = torch.stack(batch_item)
        else:
            pass
        output_batch[item_name] = batch_item


    return output_batch


class ObjectDetectionModule(BaseModule):
    def __init__(self, config=None):
        super(ObjectDetectionModule, self).__init__(config)

        anchor_sizes = [tuple(e) for e in config.model.anchor_set.anchor_sizes]
        anchor_set = AnchorSet(anchor_sizes, k_grid=config.model.anchor_set.k_grid)

        self.model = MyNet32(
            n_layers_backbone=config.model.n_layers_backbone,
            n_layers_clf_head=config.model.n_layers_clf_head,
            n_layers_reg_head=config.model.n_layers_reg_head,
            anchor_sizes=anchor_sizes,
            anchors=anchor_set.list_mnistboxes
        )

        self.ds_train = ImagesDataset(
            "train",
            config.dataset.train.size,
            config.dataset.iou_threshold,
            anchors=anchor_set.list_mnistboxes
        )
        self.ds_val = ImagesDataset(
            "val",
            None,
            config.dataset.iou_threshold,
            anchors=anchor_set.list_mnistboxes
        )
        self.ds_test = self.ds_val

        self.anchors = anchor_set.list_mnistboxes
        self.collate_fn = custom_collate_fn

        self.save_hyperparameters(config)

    def training_step(self, batch, batch_idx):
        output = self.model(batch['canvas'])

        loss_batch = 0
        n_anchors_batch = 0
        batch_size = output.classification_output.shape[0]
        for idx_sample in range(batch_size):
            matched_anchors = batch['matched_anchors'][idx_sample].tolist()
            # Trim -1's
            matched_anchors = [e for e in matched_anchors if e != -1]

            clf_target = batch['classification_target'][idx_sample][matched_anchors]
            clf_output = output.classification_output[idx_sample][matched_anchors]

            focal_loss= torchvision.ops.sigmoid_focal_loss(clf_output, clf_target, reduction="sum")

            boxreg_target = batch['box_regression_target'][idx_sample][matched_anchors]
            boxreg_output = output.box_regression_output[idx_sample][matched_anchors]

            smooth_l1_loss = F.smooth_l1_loss(boxreg_output, boxreg_target, reduction="sum")

            total_loss = focal_loss + smooth_l1_loss

            loss_batch += total_loss
            n_anchors_batch += len(matched_anchors)

        # Average loss over number of anchors
        loss_batch = loss_batch / n_anchors_batch

        return loss_batch

    def validation_step(self, batch, batch_idx):
        outputs = self.model(batch['canvas'])
        batch['classification_output'] = outputs.classification_output
        batch['box_regression_output'] = outputs.box_regression_output
        return batch

    def validation_epoch_end(self, outputs):
        classification_output = torch.cat([batch['classification_output'] for batch in outputs]).cpu()
        box_regression_output = torch.cat([batch['box_regression_output'] for batch in outputs]).cpu()
        gt_boxes = sum([batch['boxes'] for batch in outputs], [])

        for threshold, metric_name in zip(
            [0.5, 0.6, 0.7],
            ["Val/AccC5", "Val/AccC6", "Val/AccC7"]
        ):

            acc = get_metrics_batch(
                anchors=self.anchors,
                classification_output=classification_output,
                box_regression_output=box_regression_output,
                gt_boxes=gt_boxes,
                iou_threshold=0.5,
                confidence_threshold=threshold
            )
            acc = acc['recall']
            self.log(metric_name, acc)
            if metric_name == "Val/AccC7":
                print(f"\n {metric_name} = {acc:.2f}")

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        chosen_output = DigitDetectionModelOutput(
            self.anchors,
            outputs[0]['classification_output'][4].cpu(),
            outputs[0]['box_regression_output'][4].cpu(),
        )
        fig = plot_predictions(
            model_output=chosen_output,
            canvas_image=outputs[0]['canvas'][4].squeeze().cpu().numpy(),
            limit=100
        )

        # Log the plot to wandb
        wandb.log({"Predictions": wandb.Image(fig)})

        # Close the plot to avoid memory issues
        plt.close(fig)
