from time import time

import torch
import omegaconf
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


def train(config: omegaconf.DictConfig):
    n_tokens = 16
    max_count = 9

    model = EncoderModel(
        n_tokens, config.hidden_dim, config.ff_dim,
        config.n_layers, config.n_heads, output_dim=(max_count + 1)
    )
    _train_model(model, config.lr, config.num_steps, config.batch_size, n_tokens, max_count)

config = {
    "hidden_dim": 128,
    "ff_dim": 256,
    "n_heads": 8,
    "n_layers": 2,
    "batch_size": 7,
    "lr": 0.001,
    "num_steps": 1000
}

# Convert to DictConfig
config = omegaconf.OmegaConf.create(config)

train(config)

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
