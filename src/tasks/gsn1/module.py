from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.metrics import confusion_matrix
import seaborn as sns

from src.common.module import BaseModule
from src.tasks.gsn1.arch import ShapeClassificationNet
from src.tasks.gsn1.dataset import ImagesDataset
from src.tasks.gsn1.metrics import bcewithlogits_multilabel, convert_topk_to_binary, chunkwise_softmax_2d_and_reshape, loss_counting

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

        # TODO to powinno logowac do wandb, a nie na ekran
        # self._plot_confusion_matrix(preds_binary, targets)

    def _plot_confusion_matrix(self, preds, targets):
        # Ensure preds and targets are NumPy arrays
        preds = preds.cpu().numpy() if hasattr(preds, 'cpu') else preds
        targets = targets.cpu().numpy() if hasattr(targets, 'cpu') else targets

        # Generate all combinations of two indices from six classes
        class_indices = list(range(6))
        index_pairs = list(combinations(class_indices, 2))  # List of 15 tuples

        # Create a mapping from index pairs to class labels (integers 0 to 14)
        pair_to_label = {pair: idx for idx, pair in enumerate(index_pairs)}
        # Mapping from label to index pairs (for label names)
        label_to_pair = {idx: pair for idx, pair in enumerate(index_pairs)}

        # Symbolic representation of each class
        symbols = ['⬛', '●', '▲', '▶', '▼', '◀']
        # Create label names using the symbolic representations
        label_names = []
        for pair in index_pairs:
            label_names.append(f"{symbols[pair[0]]} {symbols[pair[1]]}")

        # Function to map a binary vector to a class label
        def binary_vector_to_label(row):
            indices = np.where(row == 1)[0]
            indices = tuple(sorted(indices))
            return pair_to_label[indices]

        # Map preds and targets to class labels
        preds_labels = np.apply_along_axis(binary_vector_to_label, 1, preds)
        targets_labels = np.apply_along_axis(binary_vector_to_label, 1, targets)

        # Compute the confusion matrix
        cm = confusion_matrix(targets_labels, preds_labels, labels=range(len(label_names)))

        # Plotting the confusion matrix using seaborn
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=label_names, yticklabels=label_names)
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        plt.title('Confusion Matrix')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()



class CountsModule(BaseModule):
    def __init__(self, config=None):
        super(CountsModule, self).__init__(config)

        self.model = ShapeClassificationNet(
            out_features=60,
            input_shape=[1, 28, 28],
            **config.model
        )

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
        print(f"\n {stage}/Acc = {acc:.2f}")
        self.log(f"{stage}/Acc", acc)


class CountsEncodedModule(BaseModule):
    def __init__(self, config=None):
        super(CountsEncodedModule, self).__init__(config)

        self.model = ShapeClassificationNet(
            out_features=135,
            input_shape=[1, 28, 28],
            **config.model
        )
        self.loss = nn.CrossEntropyLoss()

        self.ds_train = ImagesDataset(config, "train")
        self.ds_val = ImagesDataset(config, "val")
        self.ds_test = ImagesDataset(config, "test")

        self.save_hyperparameters(config)

    def training_step(self, batch, batch_idx):
        x, targets = batch['x'], batch['y_counts_encoded']
        logits = self.model(x)
        loss = self.loss(logits, targets)
        return loss

    def validation_step(self, batch, batch_idx):
        x, targets = batch['x'], batch['y_counts_encoded']
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

        acc = (logits.argmax(dim=1) == targets).float().mean()
        print(f"\n {stage}/Acc = {acc:.2f}")
        self.log(f"{stage}/Acc", acc)