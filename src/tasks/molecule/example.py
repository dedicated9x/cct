#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Starting code showing how to glue together components in this repository and run a single epoch of training.

Note: you are not obliged to use any part of this code
"""
from typing import Tuple, Callable

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

# Import dataset, featurization and model
from src_ import DEVICE
from src_.data.conditions_prediction_dataset import ConditionsPredictionToyTask
from src_.data.dataset import Dataset
from src_.featurization.gat_featurizer import GatGraphFeaturizer
from src_.featurization.reaction_featurizer import ReactionFeaturizer
from src_.models.gat import GAT


def prepare_batch_generator(dataset: Dataset, featurizer: ReactionFeaturizer, batch_size: int) ->\
        Tuple[Callable, int]:
    data_x = featurizer.load(dataset.feat_dir)
    meta_data = dataset.load_metadata()
    num_batches = int(np.ceil(data_x['atom'].shape[0] / batch_size))

    def gen_batches():
        for batch_ind, data_ind in enumerate(range(0, data_x['atom'].shape[0], batch_size)):
            batch_dict = {col: data_x[col][data_ind: data_ind + batch_size] for col in data_x.keys()}
            X = featurizer.unpack(batch_dict)
            y = meta_data['ortho_lithiation'].iloc[data_ind: data_ind + batch_size].astype("float32").values
            y = torch.tensor(y).to(DEVICE)
            yield X, y, batch_ind
            if batch_ind + 1 == num_batches:
                break

    return gen_batches, num_batches


def run_epoch(train_mode, batch_gen, num_batches, epoch, model, optimizer, loss_fnc):
    epoch_metrics = {
        'epoch': epoch,
        'loss': 0.0,
        'n_batches': 0.0
    }

    model = model.to(DEVICE)

    if train_mode:
        model.train()
    else:
        model.eval()

    for data, target, batch_ind in tqdm(batch_gen, desc=f'Training epoch {epoch}', total=num_batches):
        if train_mode:
            optimizer.zero_grad()

        forward_call = model.forward
        output = forward_call(data)
        loss = loss_fnc(output, target)
        total_loss = loss.mean()

        if train_mode:
            total_loss.backward()
            optimizer.step()

        epoch_metrics['loss'] += float(total_loss.cpu().detach().numpy())
        epoch_metrics['n_batches'] += 1

    epoch_metrics['loss'] = epoch_metrics['loss'] / epoch_metrics['n_batches']
    return epoch_metrics


if __name__ == "__main__":
    dataset = ConditionsPredictionToyTask()
    featurizer = GatGraphFeaturizer(n_jobs=1)
    batch_size = 128
    batch_gen_fn, num_batches = prepare_batch_generator(dataset, featurizer, batch_size)

    model = GAT()
    optim = torch.optim.Adam(lr=0.001, params=model.parameters())
    loss_fnc = nn.BCELoss(reduction='none')

    print("Started training (can take a few minutes)")
    metrics = run_epoch(train_mode=True,
                        batch_gen=batch_gen_fn(),
                        num_batches=num_batches,
                        epoch=0,
                        model=model,
                        optimizer=optim,
                        loss_fnc=loss_fnc)

    print("Metrics after one epoch")
    print(metrics)
