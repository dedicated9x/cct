from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common.module import BaseModule
from src.tasks.gsn1.arch import ShapeClassificationNet
from src.tasks.gsn1.dataset import ImagesDataset
from src.tasks.gsn1.metrics import bcewithlogits_multilabel, convert_topk_to_binary

class Gsn1BaseModule(BaseModule):
    def __init__(self, config=None):
        super(Gsn1BaseModule, self).__init__(config)

        # TODO wrzuc tu config
        self.model = ShapeClassificationNet()
        self.loss_train = nn.CrossEntropyLoss()

        self.ds_train = ImagesDataset(config, "train")
        self.ds_val = ImagesDataset(config, "val")
        self.ds_test = ImagesDataset(config, "test")

        self.save_hyperparameters(config)

    def training_step(self, batch, batch_idx):
        x, targets = batch['x'], batch['y_labels']
        logits = self.model(x)
        loss = bcewithlogits_multilabel(logits, targets)
        return loss

    def validation_step(self, batch, batch_idx):
        x, targets = batch['x'], batch['y_labels']
        logits = self.model(x)
        return {"logits": logits, "targets": targets}

    def validation_epoch_end(self, outputs):
        logits = torch.cat([batch['logits'] for batch in outputs])
        targets = torch.cat([batch['targets'] for batch in outputs])

        preds = torch.sigmoid(logits)
        preds_binary = convert_topk_to_binary(preds, 2)

        acc = (preds_binary.int() == targets).all(dim=1).float().mean()
        print(f"\n Val/Acc1 = {acc:.2f}")
        self.log("Val/Acc1", acc)

    # TODO sprawdzic, czy self.validation_step() zadziala
    def test_step(self, batch, batch_idx):
        x, targets = batch['x'], batch['y_labels']
        logits = self.model(x)
        return {"logits": logits, "targets": targets}

    def test_epoch_end(self, outputs):
        logits = torch.cat([batch['logits'] for batch in outputs])
        targets = torch.cat([batch['targets'] for batch in outputs])

        preds = torch.sigmoid(logits)
        preds_binary = convert_topk_to_binary(preds, 2)

        acc = (preds_binary.int() == targets).all(dim=1).float().mean()
        print(f"\n Test/Acc1 = {acc:.2f}")
        self.log("Test/Acc1", acc)

