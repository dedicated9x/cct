from pathlib import Path
import torch
import numpy as np

from src.common.module import BaseModule
from src.arch import ShapeClassificationNet
from src.dataset import ImagesDataset
from src.metrics import bcewithlogits_multilabel, convert_topk_to_binary

class ShapesModule(BaseModule):
    def __init__(self, config=None):
        super(ShapesModule, self).__init__(config)

        self.model = ShapeClassificationNet(
            out_features=6,
            input_shape=[1, 28, 28],
            **config.model
        )

        self.ds_train = ImagesDataset(config, "train")
        self.ds_val = ImagesDataset(config, "val")
        self.ds_test = ImagesDataset(config, "test")

        self.save_hyperparameters(config)

    def training_step(self, batch, batch_idx):
        x, targets = batch['x'], batch['y_shapes']
        logits = self.model(x)
        loss = bcewithlogits_multilabel(logits, targets)
        return loss

    def validation_step(self, batch, batch_idx):
        x, targets = batch['x'], batch['y_shapes']
        logits = self.model(x)
        return {"logits": logits, "targets": targets}

    def validation_epoch_end(self, outputs):
        logits = torch.cat([batch['logits'] for batch in outputs])
        targets = torch.cat([batch['targets'] for batch in outputs])

        preds = torch.sigmoid(logits)
        preds_binary = convert_topk_to_binary(preds, 2)

        acc = (preds_binary.int() == targets).all(dim=1).float().mean()
        print(f"\n Val/Acc = {acc:.2f}")
        self.log(f"Val/Acc", acc)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        logits = torch.cat([batch['logits'] for batch in outputs])
        targets = torch.cat([batch['targets'] for batch in outputs])

        preds = torch.sigmoid(logits)
        preds_binary = convert_topk_to_binary(preds, 2)

        acc = (preds_binary.int() == targets).all(dim=1).float().mean()
        print(f"\n Test/Acc = {acc:.2f}")
        self.log(f"Test/Acc", acc)
