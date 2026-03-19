"""
evaluate.py

Loads saved checkpoints for all three models,
compares them on the validation set, and produces:
  - comparison table (accuracy, F1, param count)
  - loss curves plot
  - confusion matrices plot
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import (
    f1_score, confusion_matrix, ConfusionMatrixDisplay
)

from src.models import ConcatMLP, BilinearFusion, CrossAttentionFusion, CrossAttentionFusionV2
from src.dataset import VQADataset
from src.train import forward
from src.utils import get_device, get_logger, set_seed

logger = get_logger("evaluate")


def count_parameters(model):
    """Count trainable parameters.
    
    WHY? A fairer comparison considers both accuracy AND
    model complexity. A bigger model should score higher
    to justify its extra parameters.
    """
    return sum(p.numel() for p in model.parameters()
               if p.requires_grad)


@torch.no_grad()
def evaluate_model(model, loader, device):
    """Run model on val set, return predictions and labels."""
    model.eval()
    all_preds = []
    all_labels = []

    for batch in loader:
        label = batch["label"].to(device)
        logits = forward(model, batch, device)
        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(label.cpu().numpy())

    return np.array(all_preds), np.array(all_labels)


def plot_loss_curves(histories, results_dir):
    """Plot train and val loss curves for all three models."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Training Loss Curves", fontsize=14, fontweight="bold")

    for ax, (name, history) in zip(axes, histories.items()):
        epochs = range(1, len(history["train_loss"]) + 1)
        ax.plot(epochs, history["train_loss"], label="train")
        ax.plot(epochs, history["val_loss"],   label="val")
        ax.set_title(name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = Path(results_dir) / "loss_curves.png"
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"Saved loss curves → {path}")


def plot_confusion_matrices(results, results_dir):
    """Plot confusion matrices for all three models side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Confusion Matrices", fontsize=14, fontweight="bold")

    for ax, (name, (preds, labels)) in zip(axes, results.items()):
        cm = confusion_matrix(labels, preds)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["No", "Yes"]
        )
        disp.plot(ax=ax, colorbar=False)
        ax.set_title(name)

    plt.tight_layout()
    path = Path(results_dir) / "confusion_matrices.png"
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"Saved confusion matrices → {path}")


def run_evaluation(config):
    set_seed(config["training"]["seed"])
    device = get_device()

    # load dataset — same split as training
    dataset = VQADataset(config["data"]["embeddings_path"])
    n_train = int(len(dataset) * config["data"]["train_split"])
    n_val   = len(dataset) - n_train
    _, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(
            config["training"]["seed"]
        )
    )

    val_loader = DataLoader(
        val_set,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
    )

    # model definitions — must match training
    model_defs = {
        "concat_mlp": ConcatMLP(
            embed_dim=config["model"]["embed_dim"],
            hidden_dim=config["model"]["hidden_dim"],
            num_classes=config["model"]["num_classes"],
            dropout=config["model"]["dropout"],
        ),
        "bilinear": BilinearFusion(
            embed_dim=config["model"]["embed_dim"],
            hidden_dim=config["model"]["hidden_dim"],
            num_classes=config["model"]["num_classes"],
            dropout=config["model"]["dropout"],
        ),
        "cross_attention": CrossAttentionFusion(
            embed_dim=config["model"]["embed_dim"],
            num_heads=config["model"]["num_heads"],
            dropout=config["model"]["dropout"],
            num_classes=config["model"]["num_classes"],
        ),
        "cross_attention_v2": CrossAttentionFusionV2(
            embed_dim=config["model"]["embed_dim"],
            num_heads=config["model"]["num_heads"],
            dropout=config["model"]["dropout"],
            num_classes=config["model"]["num_classes"],
            num_layers=config["model"]["num_layers"],
        ),
    }

    ckpt_dir     = Path(config["training"]["checkpoint_dir"])
    results_dir  = Path(config["eval"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    eval_results  = {}
    histories     = {}
    summary       = []

    for name, model in model_defs.items():
        ckpt_path = ckpt_dir / f"{name}_best.pt"
        if not ckpt_path.exists():
            logger.warning(f"No checkpoint found for {name}, skipping")
            continue

        # load checkpoint
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        model = model.to(device)

        # evaluate
        preds, labels = evaluate_model(model, val_loader, device)
        acc = (preds == labels).mean()
        f1  = f1_score(labels, preds, average="binary")
        params = count_parameters(model)

        eval_results[name] = (preds, labels)

        # load history
        hist_path = results_dir / f"{name}_history.json"
        if hist_path.exists():
            with open(hist_path) as f:
                histories[name] = json.load(f)

        summary.append({
            "model":      name,
            "val_acc":    round(float(acc), 4),
            "f1":         round(float(f1), 4),
            "params":     params,
            "best_epoch": ckpt["epoch"],
        })

        logger.info(
            f"{name:20s} | acc: {acc:.4f} | "
            f"f1: {f1:.4f} | params: {params:,}"
        )

    # print comparison table
    print("\n" + "="*60)
    print(f"{'Model':<20} {'Acc':>8} {'F1':>8} {'Params':>12}")
    print("="*60)
    for row in sorted(summary, key=lambda x: x["val_acc"], reverse=True):
        print(
            f"{row['model']:<20} "
            f"{row['val_acc']:>8.4f} "
            f"{row['f1']:>8.4f} "
            f"{row['params']:>12,}"
        )
    print("="*60)

    # save plots
    if histories:
        plot_loss_curves(histories, results_dir)
    if eval_results:
        plot_confusion_matrices(eval_results, results_dir)

    # save summary to json
    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved summary.json")


if __name__ == "__main__":
    import yaml
    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)
    run_evaluation(config)