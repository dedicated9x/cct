import torch
from torch import nn
from sklearn.metrics import roc_auc_score

from src.common.module import BaseModule
from src_.models.gat import GAT
from src.tasks.molecule.dataset import OrthoLithiationDataset


class MoleculeModule(BaseModule):
    def __init__(self, config=None):
        super(MoleculeModule, self).__init__(config)

        self.model = GAT()

        self.ds_train = OrthoLithiationDataset(config, "train")
        self.ds_val = OrthoLithiationDataset(config, "val")

        self.save_hyperparameters(config)

    def training_step(self, batch, batch_idx):
        loss_fnc = nn.BCELoss(reduction='none')

        data = {
            "n_nodes": batch['n_nodes'].cpu().numpy(),
            "atom": batch['atom'],
            "bond": batch['bond']
        }
        target = batch['y']

        forward_call = self.model.forward
        output = forward_call(data)
        loss = loss_fnc(output, target)
        total_loss = loss.mean()

        return total_loss

    def validation_step(self, batch, batch_idx):
        data = {
            "n_nodes": batch['n_nodes'].cpu().numpy(),
            "atom": batch['atom'],
            "bond": batch['bond']
        }
        forward_call = self.model.forward
        output = forward_call(data)

        batch['logit'] = output
        return batch

    def validation_epoch_end(self, outputs):
        logits = torch.cat([batch['logit'] for batch in outputs])
        targets = torch.cat([batch['y'] for batch in outputs])

        # Obliczenie accuracy
        predictions = (logits > 0.5).float()
        accuracy = (predictions == targets).float().mean().item()

        # Obliczenie AUROC
        # Konwersja tensorów na CPU, aby użyć funkcji sklearn
        targets_cpu = targets.cpu().numpy()
        logits_cpu = logits.cpu().numpy()
        auroc = roc_auc_score(targets_cpu, logits_cpu)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUROC: {auroc:.4f}")

        self.log(f"Val/Acc", accuracy)


