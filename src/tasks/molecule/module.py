from src.common.module import BaseModule
from src_.models.gat import GAT
from src.tasks.molecule.dataset import OrthoLithiationDataset


class MoleculeModule(BaseModule):
    def __init__(self, config=None):
        super(MoleculeModule, self).__init__(config)

        self.model = GAT()

        # TODO inny lr

        self.ds_train = OrthoLithiationDataset(config, "train")
        self.ds_val = OrthoLithiationDataset(config, "val")

        self.save_hyperparameters(config)

    def training_step(self, batch, batch_idx):
        a = 2
        # logits = self.model(batch['image'])
        #
        # loss_fn = nn.CrossEntropyLoss()
        # loss = loss_fn(logits, batch['label'])
        # return loss

    def validation_step(self, batch, batch_idx):
        a = 2
        # logits = self.model(batch['image'])
        # batch['logit'] = logits
        # return batch

    def validation_epoch_end(self, outputs):
        a = 2
        # logits = torch.cat([batch['logit'] for batch in outputs])
        # targets = torch.cat([batch['label'] for batch in outputs])
        #
        # acc = (logits.argmax(dim=1) == targets).float().mean()
        # print(f"\n Val/Acc = {acc:.2f}")
        # self.log(f"Val/Acc", acc)


