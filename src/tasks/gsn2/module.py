import torch
import torch.nn.functional as F
import torchvision

from src.common.module import BaseModule
from src.tasks.gsn2.anchor_set import AnchorSet
from src.tasks.gsn2.arch import MyNet32
from src.tasks.gsn2.dataset import ImagesDataset


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
            config.shared.iou_threshold,
            anchors=anchor_set.list_mnistboxes
        )
        self.ds_val = ImagesDataset(
            "val",
            None,
            config.shared.iou_threshold,
            anchors=anchor_set.list_mnistboxes
        )

        self.save_hyperparameters(config)

        # TODO remove
        self.counter = 0

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
        return batch_idx
        # x, targets = batch['x'], batch['y_shapes']
        # logits = self.model(x)
        # return {"logits": logits, "targets": targets}

    def validation_epoch_end(self, outputs):
        self.counter += 1
        self.log(f"Val/Acc", self.counter / 100)
        # logits = torch.cat([batch['logits'] for batch in outputs])
        # targets = torch.cat([batch['targets'] for batch in outputs])
        #
        # preds = torch.sigmoid(logits)
        # preds_binary = convert_topk_to_binary(preds, 2)
        #
        # acc = (preds_binary.int() == targets).all(dim=1).float().mean()
        # print(f"\n Val/Acc = {acc:.2f}")
        # self.log(f"Val/Acc", acc)

