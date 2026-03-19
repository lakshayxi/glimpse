"""
train_lora.py

Training loop for LoRA fine-tuning experiment.
Key differences from train.py:
  1. CLIP runs live every batch (not cached embeddings)
  2. Two optimizer param groups — LoRA params get small LR,
     fusion head gets normal LR
  3. Images tokenized inside the loop
"""

import json
from pathlib import Path

import clip
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.lora import inject_lora_into_clip, get_lora_params
from src.dataset_finetune import VQAFinetuneDataset
from src.utils import get_device, get_logger, set_seed

logger = get_logger("train_lora")


def collate_fn(batch):
    """Custom collate to handle variable-length question strings.
    
    WHY custom collate?
    Default PyTorch collate can't batch raw strings.
    We keep questions as a list and tokenize them inside the loop
    using CLIP's tokenizer.
    """
    return {
        "image":    torch.stack([b["image"] for b in batch]),
        "question": [b["question"] for b in batch],
        "label":    torch.stack([b["label"] for b in batch]),
    }


def train_epoch_lora(clip_model, fusion_model, loader,
                     optimizer, criterion, device):
    clip_model.train()
    fusion_model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in loader:
        images   = batch["image"].to(device)
        questions = batch["question"]
        labels   = batch["label"].to(device)

        # tokenize questions
        tokens = clip.tokenize(questions, truncate=True).to(device)

        # run CLIP live — gradients flow through LoRA layers
        # WHY no torch.no_grad()? We need gradients for LoRA backprop
        image_feat = clip_model.encode_image(images).float()  # (B, 512)
        text_feat  = clip_model.encode_text(tokens).float()   # (B, 512)

        # forward through fusion head
        logits = fusion_model(image_feat, text_feat)
        loss   = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()

        # gradient clipping — important for stable transformer training
        torch.nn.utils.clip_grad_norm_(
            list(clip_model.parameters()) +
            list(fusion_model.parameters()),
            max_norm=1.0
        )

        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


@torch.no_grad()
def eval_epoch_lora(clip_model, fusion_model, loader, criterion, device):
    clip_model.eval()
    fusion_model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for batch in loader:
        images    = batch["image"].to(device)
        questions = batch["question"]
        labels    = batch["label"].to(device)

        tokens     = clip.tokenize(questions, truncate=True).to(device)
        image_feat = clip_model.encode_image(images).float()
        text_feat  = clip_model.encode_text(tokens).float()

        logits = fusion_model(image_feat, text_feat)
        loss   = criterion(logits, labels)

        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


def train_lora(fusion_model, config, fusion_name):
    set_seed(config["training"]["seed"])
    device = get_device()
    logger.info(f"LoRA training {fusion_name} on {device}")

    # load CLIP and inject LoRA
    logger.info("Loading CLIP and injecting LoRA...")
    clip_model, preprocess = clip.load("ViT-B/32", device="cpu")
    clip_model = inject_lora_into_clip(
        clip_model, rank=8, alpha=16.0, num_blocks=4
    )
    clip_model = clip_model.to(device)

    # build dataset
    dataset = VQAFinetuneDataset(
        questions_path=config["data"]["questions_path"],
        annotations_path=config["data"]["annotations_path"],
        images_dir=config["data"]["images_dir"],
        preprocess=preprocess,
        answer_type=config["data"]["answer_type"],
    )

    n_train = int(len(dataset) * config["data"]["train_split"])
    n_val   = len(dataset) - n_train
    train_set, val_set = random_split(dataset, [n_train, n_val])

    # WHY smaller batch size?
    # Running CLIP live uses much more memory than cached embeddings
    batch_size = min(64, config["training"]["batch_size"])

    train_loader = DataLoader(
        train_set, batch_size=batch_size,
        shuffle=True, collate_fn=collate_fn, num_workers=0
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size,
        shuffle=False, collate_fn=collate_fn, num_workers=0
    )

    fusion_model = fusion_model.to(device)
    criterion    = nn.CrossEntropyLoss()

    # TWO optimizer param groups — different learning rates
    # WHY? LoRA params are new and need careful small updates.
    # Fusion head params train normally.
    lora_params   = get_lora_params(clip_model)
    fusion_params = list(fusion_model.parameters())

    optimizer = torch.optim.AdamW([
        {"params": lora_params,   "lr": config["training"]["encoder_lr"]},
        {"params": fusion_params, "lr": config["training"]["lr"]},
    ], weight_decay=config["training"]["weight_decay"])

    best_val_loss    = float("inf")
    best_val_acc     = 0
    epochs_no_improve = 0
    patience         = config["training"]["early_stopping_patience"]
    history          = {"train_loss": [], "val_loss": [],
                        "train_acc":  [], "val_acc":  []}

    for epoch in range(1, config["training"]["epochs"] + 1):
        train_loss, train_acc = train_epoch_lora(
            clip_model, fusion_model, train_loader,
            optimizer, criterion, device
        )
        val_loss, val_acc = eval_epoch_lora(
            clip_model, fusion_model, val_loader,
            criterion, device
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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc  = val_acc
            epochs_no_improve = 0
            ckpt_dir = Path(config["training"]["checkpoint_dir"])
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            # save both fusion head and LoRA weights
            torch.save({
                "epoch":        epoch,
                "fusion_state": fusion_model.state_dict(),
                "lora_state":   {k: v for k, v in clip_model.state_dict().items()
                                 if "lora" in k},
                "val_acc":      val_acc,
            }, ckpt_dir / f"lora_{fusion_name}_best.pt")
            logger.info(f"  ✓ saved best checkpoint (val_loss={val_loss:.4f})")
        else:
            epochs_no_improve += 1
            logger.info(f"  no improvement for {epochs_no_improve} epoch(s)")

        if epochs_no_improve >= patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    results_dir = Path(config["eval"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / f"lora_{fusion_name}_history.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"Done. Best val acc: {best_val_acc:.4f}")
    return history