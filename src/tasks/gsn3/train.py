import matplotlib.pyplot as plt
import numpy as np
from time import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src.tasks.gsn3.dataset import DEVICE, get_single_example, OUTPUT_DIM, N_TOKENS
from src.tasks.gsn3.arch import EncoderModel

TEST_SIZE = 128

test_examples = [get_single_example() for i in range(TEST_SIZE)]

# Transpositions are used, because the convention in PyTorch is to represent
# sequence tensors as <seq_len, batch_size> instead of <batch_size, seq_len>.
test_X = torch.tensor([x[0] for x in test_examples],
                      device=DEVICE).transpose(0, 1)
test_Y = torch.tensor([x[1] for x in test_examples],
                      device=DEVICE).transpose(0, 1)


def train_model(model, lr, num_steps, batch_size):
    model.to(DEVICE)

    start_time = time()
    accs = []

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for step in range(num_steps):
        batch_examples = [get_single_example() for i in range(batch_size)]

        batch_X = torch.tensor([x[0] for x in batch_examples],
                               device=DEVICE
                               ).transpose(0, 1)
        batch_Y = torch.tensor([x[1] for x in batch_examples],
                               device=DEVICE).transpose(0, 1)

        model.train()
        model.zero_grad()
        logits = model(batch_X)
        loss = loss_function(logits.reshape(-1, OUTPUT_DIM), batch_Y.reshape(-1))
        loss.backward()
        optimizer.step()

        if step % (num_steps // 100) == 0 or step == num_steps - 1:
            # Printing a summary of the current state of training every 1% of steps.
            model.eval()
            predicted_logits = model.forward(test_X).reshape(-1, OUTPUT_DIM)
            test_acc = (
                    torch.sum(torch.argmax(predicted_logits, dim=-1) == test_Y.reshape(-1))
                    / test_Y.reshape(-1).shape[0])
            print('step', step, 'out of', num_steps)
            print('loss train', float(loss))
            print('accuracy test', float(test_acc))
            print()
            accs.append(test_acc)
    print('\nTRAINING TIME:', time() - start_time)
    model.eval()
    return accs

# TODO: change those placeholder parameters
HIDDEN_DIM = 1  # change
FF_DIM = 1  # change
N_HEADS = 1  # change
N_LAYERS = 1  # change

BATCH_SIZE = 3  # change
LR = 0.01  # change
NUM_STEPS = 200  # change

# Let's choose appropriate hyperparameters:
HIDDEN_DIM = 128
FF_DIM = 256
N_HEADS = 8
N_LAYERS = 2
BATCH_SIZE = 64
LR = 0.001
NUM_STEPS = 1000

model = EncoderModel(N_TOKENS, HIDDEN_DIM, FF_DIM, OUTPUT_DIM, N_LAYERS, N_HEADS)
accs = train_model(model, LR, NUM_STEPS, BATCH_SIZE)
plt.plot([i * NUM_STEPS/(len(accs)-1) for i in range(len(accs))], accs)
plt.xlabel('Steps')
plt.ylabel('Test Acc')
plt.show()