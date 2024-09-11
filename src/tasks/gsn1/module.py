from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common.module import BaseModule
from src.tasks.gsn1.arch import ShapeClassificationNet
from src.tasks.gsn1.dataset import ImagesDataset


class Gsn1BaseModule(BaseModule):
    def __init__(self, config=None):
        super(Gsn1BaseModule, self).__init__(config)

        # TODO wrzuc tu config
        self.model = ShapeClassificationNet()
        # TODO zbadaj dalsze losy ten loss_train
        self.loss_train = nn.CrossEntropyLoss()

        self.ds_train = ImagesDataset(config, "train")
        self.ds_val = ImagesDataset(config, "val")
        self.ds_test = ImagesDataset(config, "test")

        self.save_hyperparameters(config)

    def training_step(self, batch, batch_idx):
        a = 2
        raise NotImplementedError
        # x, y = batch['x'], batch['y']
        # y_hat = self(x)
        # y = F.one_hot(y.to(torch.int64), num_classes=17).to(torch.float32)
        # loss = self.loss_train(y_hat, y)
        # return loss

    def validation_epoch_end(self, outputs):
        a = 2
        raise NotImplementedError
        # y_hat = torch.cat([batch['y_hat'] for batch in outputs])
        # y = torch.cat([batch['y'] for batch in outputs])
        # acc = (y_hat.argmax(dim=1) == y).to(torch.float32).mean()
        # print(f"\n Val/Acc1 = {acc:.2f}")
        # self.log("Val/Acc1", acc)

    def test_epoch_end(self, outputs):
        a = 2
        raise NotImplementedError
        # y_hat = torch.cat([batch['y_hat'] for batch in outputs])
        # y = torch.cat([batch['y'] for batch in outputs])
        # acc = (y_hat.argmax(dim=1) == y).to(torch.float32).mean()
        # self.log("Test/Acc1", acc)

