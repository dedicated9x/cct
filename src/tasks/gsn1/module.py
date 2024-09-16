from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common.module import BaseModule
from src.tasks.gsn1.arch import ShapeClassificationNet
from src.tasks.gsn1.dataset import ImagesDataset
from src.tasks.gsn1.metrics import bcewithlogits_multilabel, convert_topk_to_binary, chunkwise_softmax_2d_and_reshape, loss_counting

class ShapesModule(BaseModule):
    def __init__(self, config=None):
        super(ShapesModule, self).__init__(config)

        self.model = ShapeClassificationNet(out_features=6)
        self.loss_train = nn.CrossEntropyLoss()

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
        self._compute_epoch_end_metrics(outputs, stage='Val')

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        self._compute_epoch_end_metrics(outputs, stage='Test')

    def _compute_epoch_end_metrics(self, outputs, stage):
        logits = torch.cat([batch['logits'] for batch in outputs])
        targets = torch.cat([batch['targets'] for batch in outputs])

        preds = torch.sigmoid(logits)
        preds_binary = convert_topk_to_binary(preds, 2)

        acc = (preds_binary.int() == targets).all(dim=1).float().mean()
        print(f"\n {stage}/Acc1 = {acc:.2f}")
        self.log(f"{stage}/Acc1", acc)


class CountsModule(BaseModule):
    def __init__(self, config=None):
        super(CountsModule, self).__init__(config)

        self.model = ShapeClassificationNet(out_features=60)
        self.loss_train = nn.CrossEntropyLoss()

        self.ds_train = ImagesDataset(config, "train")
        self.ds_val = ImagesDataset(config, "val")
        self.ds_test = ImagesDataset(config, "test")

        self.save_hyperparameters(config)

    def training_step(self, batch, batch_idx):
        x, targets = batch['x'], batch['y_counts']
        logits_flattened = self.model(x)

        preds = chunkwise_softmax_2d_and_reshape(x=logits_flattened, chunk_size=10,)

        loss = loss_counting(counts=targets, preds=preds, device=x.device)
        return loss

    def validation_step(self, batch, batch_idx):
        x, targets = batch['x'], batch['y_counts']
        logits_flattened = self.model(x)
        return {"logits_flattened": logits_flattened, "targets": targets}

    def validation_epoch_end(self, outputs):
        self._compute_epoch_end_metrics(outputs, stage='Val')

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        self._compute_epoch_end_metrics(outputs, stage='Test')

    def _compute_epoch_end_metrics(self, outputs, stage):
        logits_flattened = torch.cat([batch['logits_flattened'] for batch in outputs])
        targets = torch.cat([batch['targets'] for batch in outputs])

        preds = chunkwise_softmax_2d_and_reshape(x=logits_flattened, chunk_size=10, )
        preds_counts = preds.argmax(dim=2)

        acc = (preds_counts.int() == targets).all(dim=1).float().mean()
        print(f"\n {stage}/Acc1 = {acc:.2f}")
        self.log(f"{stage}/Acc1", acc)

