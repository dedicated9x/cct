from time import time

import torch
import torch.nn as nn
import torch.optim as optim

from src.tasks.gsn3.dataset import get_single_example
from src.tasks.gsn3.arch import EncoderModel


def _train_model(
        model,
        lr,
        num_steps,
        batch_size,
        n_tokens,
        max_count
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_len = 64
    test_size = 128
    output_dim = max_count + 1

    test_examples = [get_single_example(n_tokens, seq_len, max_count) for i in range(test_size)]

    test_X = torch.tensor([x[0] for x in test_examples], device=device).transpose(0, 1)
    test_Y = torch.tensor([x[1] for x in test_examples], device=device).transpose(0, 1)

    model.to(device)

    start_time = time()
    accs = []

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for step in range(num_steps):
        batch_examples = [get_single_example(n_tokens, seq_len, max_count) for i in range(batch_size)]

        batch_X = torch.tensor([x[0] for x in batch_examples], device=device).transpose(0, 1)
        batch_Y = torch.tensor([x[1] for x in batch_examples], device=device).transpose(0, 1)

        model.train()
        model.zero_grad()
        logits = model(batch_X)
        loss = loss_function(logits.reshape(-1, output_dim), batch_Y.reshape(-1))
        loss.backward()
        optimizer.step()

        if step % (num_steps // 100) == 0 or step == num_steps - 1:
            # Printing a summary of the current state of training every 1% of steps.
            model.eval()
            predicted_logits = model.forward(test_X).reshape(-1, output_dim)
            test_acc = (
                    torch.sum(torch.argmax(predicted_logits, dim=-1) == test_Y.reshape(-1))
                    / test_Y.reshape(-1).shape[0]
            )
            print('step', step, 'out of', num_steps)
            print('loss train', float(loss))
            print('accuracy test', float(test_acc))
            print()
            accs.append(test_acc)
    print('\nTRAINING TIME:', time() - start_time)
    model.eval()
    return accs

N_TOKENS = 16
MAX_COUNT = 9

# Let's choose appropriate hyperparameters:
HIDDEN_DIM = 128
FF_DIM = 256
N_HEADS = 8
N_LAYERS = 2
BATCH_SIZE = 7
LR = 0.001
NUM_STEPS = 1000

model = EncoderModel(N_TOKENS, HIDDEN_DIM, FF_DIM, N_LAYERS, N_HEADS, output_dim=(MAX_COUNT + 1))
accs = _train_model(model, LR, NUM_STEPS, BATCH_SIZE, N_TOKENS, MAX_COUNT)




# step 0 out of 1000
# loss train 2.3270153999328613
# accuracy test 0.004150390625
#
# step 10 out of 1000
# loss train 1.804653525352478
# accuracy test 0.2454833984375
#
# step 20 out of 1000
# loss train 1.8742406368255615
# accuracy test 0.227783203125
#
# step 30 out of 1000
# loss train 1.8240848779678345
# accuracy test 0.227783203125
#
# step 40 out of 1000
# loss train 1.8283302783966064
# accuracy test 0.2454833984375
