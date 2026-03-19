"""
train.py

Shared training loop for all three fusion models.
WHY shared? Because the training logic is identical across models.
The only thing that changes is which model we pass in.
"""

import os
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.dataset import VQADataset
from src.utils import get_device, get_logger, set_seed

logger = get_logger("train")


def train_epoch(model, loader, optimizer, criterion, device):
    """One full pass over the training data."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in loader:
        # move data to device
        label = batch["label"].to(device)

        # forward pass — different models need different inputs
        logits = forward(model, batch, device)

        # compute loss
        loss = criterion(logits, label)

        # zero gradients, backward, update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # track metrics
        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        correct += (preds == label).sum().item()
        total += label.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    """One full pass over the validation data. No gradient updates."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for batch in loader:
        label = batch["label"].to(device)
        logits = forward(model, batch, device)
        loss = criterion(logits, label)

        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        correct += (preds == label).sum().item()
        total += label.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy


def forward(model, batch, device):
    """Route the right inputs to the right model.
    
    WHY this function?
    ConcatMLP and Bilinear use global image features (B, 512).
    CrossAttention uses patch features (B, 196, 512).
    This function checks which model we're using and sends
    the correct inputs automatically.
    """
    model_name = model.__class__.__name__

    text_feat = batch["text_feat"].to(device)

    if model_name in ("CrossAttentionFusion", "CrossAttentionFusionV2"):
        image_feat = batch["image_patches"].to(device)
    else:
        image_feat = batch["image_global"].to(device)

    return model(image_feat, text_feat)


def train(model, config, fusion_name):
    """Full training run for one model."""

    set_seed(config["training"]["seed"])
    device = get_device()
    logger.info(f"Training {fusion_name} on {device}")

    # load dataset and split into train/val
    dataset = VQADataset(config["data"]["embeddings_path"])
    n_train = int(len(dataset) * config["data"]["train_split"])
    n_val = len(dataset) - n_train
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_set,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
    )

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
    )

    best_val_acc = 0
    best_val_loss = float("inf")
    epochs_no_improve = 0
    patience = config["training"]["early_stopping_patience"]
    history = {"train_loss": [], "val_loss": [],
               "train_acc": [], "val_acc": []}

    for epoch in range(1, config["training"]["epochs"] + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = eval_epoch(
            model, val_loader, criterion, device
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        logger.info(
            f"Epoch {epoch:02d} | "
            f"train loss: {train_loss:.4f} acc: {train_acc:.4f} | "
            f"val loss: {val_loss:.4f} acc: {val_acc:.4f}"
        )

        # save checkpoint if best val loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            epochs_no_improve = 0
            ckpt_dir = Path(config["training"]["checkpoint_dir"])
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_acc": val_acc,
            }, ckpt_dir / f"{fusion_name}_best.pt")
            logger.info(f"  ✓ saved best checkpoint (val_loss={val_loss:.4f})")
        else:
            epochs_no_improve += 1
            logger.info(f"  no improvement for {epochs_no_improve} epoch(s)")

        # early stopping
        if epochs_no_improve >= patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    # save training history
    results_dir = Path(config["eval"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / f"{fusion_name}_history.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"Done. Best val acc: {best_val_acc:.4f}")
    return history