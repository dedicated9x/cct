import torch
import pytorch_lightning as pl

from src.arch import ShapeClassificationNet
from src.dataset import ImagesDataset
from src.metrics import bcewithlogits_multilabel, convert_topk_to_binary

class ShapesModule(pl.LightningModule):
    def __init__(self, config=None):
        super(ShapesModule, self).__init__()
        self.config = config

        self.model = ShapeClassificationNet(
            out_features=6,
            input_shape=[1, 28, 28],
            **config.model
        )

        self.ds_train = ImagesDataset(config, "train")
        self.ds_val = ImagesDataset(config, "val")
        self.ds_test = ImagesDataset(config, "test")

        self.save_hyperparameters(config)
        self._val_step_outputs = []
        self._test_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.ds_train,
            batch_size=self.config.trainer.batch_size,
            shuffle=True,
            num_workers=4,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.ds_val,
            batch_size=self.config.trainer.batch_size,
            num_workers=4,
        )

    def test_dataloader(self):
        if self.config.main.is_tested:
            return torch.utils.data.DataLoader(
                self.ds_test,
                batch_size=self.config.trainer.batch_size,
                num_workers=4,
            )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.optimizer.lr)
        return optimizer


    def training_step(self, batch, batch_idx):
        x, targets = batch['x'], batch['y_shapes']
        logits = self.model(x)
        loss = bcewithlogits_multilabel(logits, targets)
        return loss

    def validation_step(self, batch, batch_idx):
        x, targets = batch['x'], batch['y_shapes']
        logits = self.model(x)
        self._val_step_outputs.append({"logits": logits.detach(), "targets": targets.detach()})

    def on_validation_epoch_end(self):
        if not self._val_step_outputs:
            return

        logits = torch.cat([batch['logits'] for batch in self._val_step_outputs])
        targets = torch.cat([batch['targets'] for batch in self._val_step_outputs])

        preds = torch.sigmoid(logits)
        preds_binary = convert_topk_to_binary(preds, 2)

        acc = (preds_binary.int() == targets).all(dim=1).float().mean()
        print(f"\n Val/Acc = {acc:.2f}")
        self.log("Val/Acc", acc, prog_bar=True)
        self._val_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        x, targets = batch['x'], batch['y_shapes']
        logits = self.model(x)
        self._test_step_outputs.append({"logits": logits.detach(), "targets": targets.detach()})

    def on_test_epoch_end(self):
        if not self._test_step_outputs:
            return

        logits = torch.cat([batch['logits'] for batch in self._test_step_outputs])
        targets = torch.cat([batch['targets'] for batch in self._test_step_outputs])

        preds = torch.sigmoid(logits)
        preds_binary = convert_topk_to_binary(preds, 2)

        acc = (preds_binary.int() == targets).all(dim=1).float().mean()
        print(f"\n Test/Acc = {acc:.2f}")
        self.log("Test/Acc", acc, prog_bar=True)
        self._test_step_outputs.clear()
