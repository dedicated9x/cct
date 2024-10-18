import torch
import torch.nn as nn
import timm

from src.common.module import BaseModule
from src.tasks.appsilon.dataset import AppsilonDataset

class AppsilonModule(BaseModule):
    def __init__(self, config=None):
        super(AppsilonModule, self).__init__(config)

        self.model = timm.create_model(
            'vit_base_patch16_224',
            pretrained=True,
            num_classes=17
        )


        self.ds_train = AppsilonDataset(config, "train")
        self.ds_val = AppsilonDataset(config, "val")
        self.ds_test = AppsilonDataset(config, "test")

        self.save_hyperparameters(config)

    def training_step(self, batch, batch_idx):
        logits = self.model(batch['image'])

        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits, batch['label'])
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self.model(batch['image'])
        batch['logit'] = logits
        return batch

    def validation_epoch_end(self, outputs):
        logits = torch.cat([batch['logit'] for batch in outputs])
        targets = torch.cat([batch['label'] for batch in outputs])

        acc = (logits.argmax(dim=1) == targets).float().mean()
        print(f"\n Val/Acc = {acc:.2f}")
        self.log(f"Val/Acc", acc)

    # def test_step(self, batch, batch_idx):
    #     return self.validation_step(batch, batch_idx)
    #
    # def test_epoch_end(self, outputs):
    #     logits = torch.cat([batch['logits'] for batch in outputs])
    #     targets = torch.cat([batch['targets'] for batch in outputs])
    #
    #     preds = torch.sigmoid(logits)
    #     preds_binary = None
    #
    #     acc = (preds_binary.int() == targets).all(dim=1).float().mean()
    #     print(f"\n Test/Acc = {acc:.2f}")
    #     self.log(f"Test/Acc", acc)
