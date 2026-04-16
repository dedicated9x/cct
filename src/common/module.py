import torch
import torch.utils.data
import pytorch_lightning as pl


class BaseModule(pl.LightningModule):
    def __init__(self, config=None):
        super(BaseModule, self).__init__()
        self.config = config

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

