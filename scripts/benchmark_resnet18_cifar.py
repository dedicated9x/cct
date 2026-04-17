#!/usr/bin/env python3
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from tqdm import tqdm
from torchvision import datasets, models, transforms


# =======================
# User-configurable params
# =======================
EPOCHS = 1
BATCH_SIZE = 128
LR = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
NUM_WORKERS = 4
SEED = 42
DATA_DIR = Path("data/cifar10")
DATA_FRACTION = 0.1


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def build_dataloaders() -> tuple[DataLoader, DataLoader]:
    train_tfms = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            ),
        ]
    )

    val_tfms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            ),
        ]
    )

    train_ds = datasets.CIFAR10(
        root=str(DATA_DIR),
        train=True,
        download=True,
        transform=train_tfms,
    )
    val_ds = datasets.CIFAR10(
        root=str(DATA_DIR),
        train=False,
        download=True,
        transform=val_tfms,
    )

    train_subset_size = max(1, int(len(train_ds) * DATA_FRACTION))
    val_subset_size = max(1, int(len(val_ds) * DATA_FRACTION))
    rng = random.Random(SEED)
    train_indices = rng.sample(range(len(train_ds)), train_subset_size)
    val_indices = rng.sample(range(len(val_ds)), val_subset_size)

    train_ds = Subset(train_ds, train_indices)
    val_ds = Subset(val_ds, val_indices)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )
    return train_loader, val_loader


def run_epoch_train(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(
        loader,
        total=len(loader),
        desc="train",
        leave=False,
    )
    for x, y in progress:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def run_epoch_val(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        running_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    return running_loss / total, correct / total


def main() -> None:
    set_seed(SEED)
    device = torch.device("cpu")

    train_loader, val_loader = build_dataloaders()

    model = models.resnet18(weights=None, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )

    total_start = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.perf_counter()

        train_loss, train_acc = run_epoch_train(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_loss, val_acc = run_epoch_val(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        epoch_time = time.perf_counter() - epoch_start
        print(
            f"epoch={epoch}/{EPOCHS} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"epoch_time_s={epoch_time:.2f}",
            flush=True,
        )

    total_time = time.perf_counter() - total_start
    print(f"total_training_time_s={total_time:.2f}", flush=True)


if __name__ == "__main__":
    main()

