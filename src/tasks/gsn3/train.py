from time import time
import torch
import omegaconf
import torch.nn as nn
import torch.optim as optim
import wandb
import datetime
from pathlib import Path
import pandas as pd

from src.tasks.gsn3.dataset import get_single_example
from src.tasks.gsn3.arch import EncoderModel


def get_test_dataset(
    n_tokens,
    max_count,
    seq_len,
    device
):
    test_size = 128

    test_examples = [get_single_example(n_tokens, seq_len, max_count) for i in range(test_size)]

    test_X = torch.tensor([x[0] for x in test_examples], device=device).transpose(0, 1)
    test_Y = torch.tensor([x[1] for x in test_examples], device=device).transpose(0, 1)

    return test_X, test_Y


def _train_model(
        model,
        lr,
        num_steps,
        batch_size,
        n_tokens,
        max_count,
        seq_len,
        device,
        test_X,
        test_Y,
        config
) -> None:
    output_dim = max_count + 1

    start_time = time()
    accs = []

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Initialize wandb and log config (hyperparameters)
    wandb.init(
        project="gsn3-training",
        reinit=True,
        config=omegaconf.OmegaConf.to_container(config, resolve=True)
    )

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
            predicted_logits = model.forward(test_X)
            test_acc = (
                    torch.sum(torch.argmax(predicted_logits.reshape(-1, output_dim), dim=-1) == test_Y.reshape(-1))
                    / test_Y.reshape(-1).shape[0]
            )
            print('step', step, 'out of', num_steps)
            print('loss train', float(loss))
            print('accuracy test', float(test_acc))
            print()

            # Log metrics to wandb
            wandb.log({"train_loss": loss.item(), "test_accuracy": test_acc.item(), "step": step})

            accs.append(test_acc)


    # Log the total training time
    wandb.log({"training_time": time() - start_time})

    # Save ckpt
    filename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".ckpt"
    path_ckpt = str(Path("/tmp") / filename)
    torch.save(model.state_dict(), path_ckpt)
    print(f"Weights saved to \"{path_ckpt}\".")

    # Finish the wandb run
    wandb.finish()

    model.eval()


def train(config: omegaconf.DictConfig):
    n_tokens = 16
    max_count = 9
    seq_len = 64


    model = EncoderModel(
        n_tokens, config.hidden_dim, config.ff_dim,
        config.n_layers, config.n_heads, output_dim=(max_count + 1)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_X, test_Y = get_test_dataset(n_tokens, max_count, seq_len, device)
    model.to(device)

    # Load the saved state_dict
    if config.ckpt_path is not None:
        model.load_state_dict(torch.load(config.ckpt_path))

    _train_model(
        model, config.lr, config.num_steps,
        config.batch_size, n_tokens, max_count, seq_len,
        device, test_X, test_Y, config
    )

if __name__ == '__main__':
    config = {
        "hidden_dim": 512,
        "ff_dim": 512,
        "n_heads": 8,
        "n_layers": 1,
        "batch_size": 64,
        "lr": 0.0001,
        "num_steps": 2000,
        # "num_steps": 200,
        # "ckpt_path": "/tmp/20241015_083926.ckpt"
        "ckpt_path": None
    }

    # Convert to DictConfig
    config = omegaconf.OmegaConf.create(config)

    train(config)
