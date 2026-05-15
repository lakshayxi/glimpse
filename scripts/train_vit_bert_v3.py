# =========================
# INSTALL (run these first in terminal):
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# pip install timm transformers pandas pillow
# =========================

import os
import json
import time
import csv
import math
from datetime import datetime
import pandas as pd
from PIL import Image
from collections import Counter

import torch
import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from transformers import AutoTokenizer, AutoModel
import timm

from torch.amp import autocast

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32  = True

# =========================
# CONFIG
# =========================
class CFG:
    IMG_SIZE        = 224
    MAX_LEN         = 32
    BATCH_SIZE      = 256
    EPOCHS          = 40
    LR              = 3e-4
    NUM_ANSWERS     = 3129
    DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
    USE_AMP         = torch.cuda.is_available()
    NUM_WORKERS     = 8 if torch.cuda.is_available() else 0
    LOG_FILE        = "training_log3.csv"
    WARMUP_EPOCHS   = 2
    LABEL_SMOOTHING = 0.1
    GRAD_CLIP       = 1.0
    ACCUMULATE      = 2

print(f"Running on: {CFG.DEVICE}")
if CFG.DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# =========================
# EPOCH LOGGER
# =========================
class EpochLogger:
    FIELDS = [
        "run_id", "epoch", "timestamp",
        "epoch_secs", "elapsed_secs",
        "loss", "val_acc", "lr", "best_acc"
    ]

    def __init__(self, path: str):
        self.path        = path
        self.run_id      = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.train_start = time.time()

        write_header = not os.path.exists(path)
        self._file   = open(path, "a", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDS)
        if write_header:
            self._writer.writeheader()
            self._file.flush()

        print(f"[EpochLogger] Logging to '{path}'  (run_id={self.run_id})")

    def log(self, epoch, epoch_secs, loss, val_acc, lr, best_acc):
        row = {
            "run_id":       self.run_id,
            "epoch":        epoch,
            "timestamp":    datetime.now().isoformat(timespec="seconds"),
            "epoch_secs":   round(epoch_secs, 2),
            "elapsed_secs": round(time.time() - self.train_start, 2),
            "loss":         round(loss,    4),
            "val_acc":      round(val_acc, 4),
            "lr":           f"{lr:.2e}",
            "best_acc":     round(best_acc, 4),
        }
        self._writer.writerow(row)
        self._file.flush()

    def close(self):
        self._file.close()

# =========================
# PATHS  ← CHANGE THESE TO YOUR LOCAL PATHS
# =========================
BASE = "/home/shlok/data/vqa_v2"

TRAIN_IMG_DIR = f"{BASE}/train2014"
VAL_IMG_DIR   = f"{BASE}/val2014"

TRAIN_Q = f"{BASE}/v2_OpenEnded_mscoco_train2014_questions.json"
TRAIN_A = f"{BASE}/v2_mscoco_train2014_annotations.json"

VAL_Q   = f"{BASE}/v2_OpenEnded_mscoco_val2014_questions.json"
VAL_A   = f"{BASE}/v2_mscoco_val2014_annotations.json"

# =========================
# LOAD DATA
# =========================
# FIX #1 — "question_id" is now included in the returned DataFrame.
# Previously this was dropped, which caused soft labels to never be
# looked up — every sample silently fell back to a hard one-hot label.
# =========================
def load_vqa(q_path, a_path, split="train"):
    with open(q_path) as f:
        questions = json.load(f)["questions"]
    with open(a_path) as f:
        annotations = json.load(f)["annotations"]

    q_df = pd.DataFrame(questions)
    a_df = pd.DataFrame(annotations)
    df   = q_df.merge(a_df, on="question_id", suffixes=("_q", "_a"))

    img_col = "image_id_q" if "image_id_q" in df.columns else "image_id_a"
    prefix  = "COCO_train2014_" if split == "train" else "COCO_val2014_"

    df["image"]  = df[img_col].apply(lambda x: f"{prefix}{str(x).zfill(12)}.jpg")
    df["answer"] = df["multiple_choice_answer"]

    # ✅ FIX: include question_id so VQADataset can look up soft labels
    return df[["image", "question", "answer", "question_id"]]


train_df = load_vqa(TRAIN_Q, TRAIN_A, "train")
val_df   = load_vqa(VAL_Q,   VAL_A,   "val")

# =========================
# SOFT SCORE LABELS (VQA v2 official metric)
# =========================
def build_soft_labels(a_path, answer_to_idx):
    with open(a_path) as f:
        annotations = json.load(f)["annotations"]

    soft = {}
    for ann in annotations:
        qid     = ann["question_id"]
        scores  = torch.zeros(len(answer_to_idx))
        counter = Counter(a["answer"] for a in ann["answers"])
        for ans, cnt in counter.items():
            if ans in answer_to_idx:
                scores[answer_to_idx[ans]] = min(cnt / 3.0, 1.0)
        soft[qid] = scores
    return soft

# =========================
# ANSWER VOCAB
# =========================
counter       = Counter(train_df["answer"])
most_common   = counter.most_common(CFG.NUM_ANSWERS)
answer_to_idx = {ans: idx for idx, (ans, _) in enumerate(most_common)}

print(f"Building soft labels for {len(train_df)} training samples...")
train_soft = build_soft_labels(TRAIN_A, answer_to_idx)
val_soft   = build_soft_labels(VAL_A,   answer_to_idx)
print("Soft labels built.")

# =========================
# TOKENIZER
# =========================
os.environ["TOKENIZERS_PARALLELISM"] = "false"
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# =========================
# TRANSFORMS
# =========================
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(CFG.IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((CFG.IMG_SIZE, CFG.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# =========================
# DATASET
# =========================
class VQADataset(Dataset):
    def __init__(self, df, img_dir, soft_labels, transform):
        self.df          = df.reset_index(drop=True)
        self.img_dir     = img_dir
        self.soft_labels = soft_labels
        self.transform   = transform

        enc = tokenizer(
            self.df["question"].tolist(),
            padding="max_length",
            truncation=True,
            max_length=CFG.MAX_LEN,
            return_tensors="pt"
        )
        self.input_ids      = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]

        # ✅ FIX: question_id column now exists, so this gives real IDs
        self.question_ids = self.df["question_id"].tolist() \
            if "question_id" in self.df.columns else [None] * len(self.df)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["image"])
        image    = Image.open(img_path).convert("RGB")
        image    = self.transform(image)

        hard_label = answer_to_idx.get(row["answer"], 0)

        qid = self.question_ids[idx]
        if qid is not None and qid in self.soft_labels:
            # ✅ FIX: now reached for every sample
            soft = self.soft_labels[qid]
        else:
            soft = torch.zeros(CFG.NUM_ANSWERS)
            soft[hard_label] = 1.0

        return {
            "image":          image,
            "input_ids":      self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "label":          torch.tensor(hard_label, dtype=torch.long),
            "soft_label":     soft,
        }

# =========================
# DATALOADER
# =========================
train_dataset = VQADataset(train_df, TRAIN_IMG_DIR, train_soft, train_transform)
val_dataset   = VQADataset(val_df,   VAL_IMG_DIR,   val_soft,   val_transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=CFG.BATCH_SIZE,
    shuffle=True,
    num_workers=CFG.NUM_WORKERS,
    pin_memory=CFG.USE_AMP,
    persistent_workers=(CFG.NUM_WORKERS > 0),
    prefetch_factor=4 if CFG.NUM_WORKERS > 0 else None,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=CFG.BATCH_SIZE,
    num_workers=CFG.NUM_WORKERS,
    pin_memory=CFG.USE_AMP,
    persistent_workers=(CFG.NUM_WORKERS > 0),
    prefetch_factor=4 if CFG.NUM_WORKERS > 0 else None,
)

# =========================
# SOFT LABEL SANITY CHECK
# Run this once before training to confirm soft labels are working.
# If you see all 1.0 → still broken.  Fractional values → correct.
# =========================
print("\n[Sanity Check] Soft label row sums (first 8 samples):")
_batch = next(iter(train_loader))
_sums  = _batch["soft_label"].sum(dim=1)[:8]
print(" ", _sums)
print("  Expected: fractional values like [1.33, 0.67, 1.0, 2.0, ...]")
print("  If all 1.0 → question_id is still not being passed correctly\n")
del _batch, _sums

# =========================
# MODEL
# =========================

# FIX #2 — CrossAttentionBlock now accepts a query_padding_mask.
# When [PAD] tokens are the query, their attention outputs are zeroed
# out before LayerNorm so they stay truly inert across all depth layers.
# Without this, [PAD] positions accumulate signal through LayerNorm and
# leak corrupted values into the next layer's KV.
# =========================
class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attn  = nn.MultiheadAttention(dim, num_heads, batch_first=True,
                                            dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.ff    = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, q, kv, key_padding_mask=None, query_padding_mask=None):
        """
        Args:
            q:                 Query tensor  [B, Tq, dim]
            kv:                Key/Value tensor [B, Tkv, dim]
            key_padding_mask:  BoolTensor [B, Tkv], True = ignore this KV token
            query_padding_mask:BoolTensor [B, Tq],  True = this query is padding
        """
        attn_out, _ = self.attn(
            q, kv, kv,
            key_padding_mask=key_padding_mask,
        )

        # ✅ FIX: zero out padded query positions after attention
        # so they don't pollute LayerNorm statistics or the next layer's KV
        if query_padding_mask is not None:
            attn_out = attn_out.masked_fill(
                query_padding_mask.unsqueeze(-1), 0.0
            )

        x = self.norm1(q + attn_out)
        out = self.norm2(x + self.ff(x))

        # ✅ FIX: zero out again after FFN+LayerNorm to keep [PAD] inert
        if query_padding_mask is not None:
            out = out.masked_fill(
                query_padding_mask.unsqueeze(-1), 0.0
            )

        return out


class GatedFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Tanh(),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )
        self.proj = nn.Linear(dim, dim)

    def forward(self, text_vec, img_vec):
        combined = torch.cat([text_vec, img_vec], dim=-1)
        alpha    = self.gate(combined)
        fused    = alpha * text_vec + (1 - alpha) * img_vec
        return self.proj(fused)


# FIX #3 — Masking is now applied symmetrically in both directions:
#   • text→image:  text [PAD] positions masked as query  (new fix)
#   • image→text:  text [PAD] positions masked as KV     (was already done)
# FIX #4 — img_pooled now uses ViT CLS token (v[:, 0]) instead of
#   mean pooling, consistent with how text_pooled uses BERT CLS (t[:, 0]).
#   Mean pooling destroyed spatial structure; CLS is a trained global summary.
# =========================
class MultiLayerCoAttention(nn.Module):
    def __init__(self, dim, depth=4, num_heads=8):
        super().__init__()

        self.text_to_img_layers = nn.ModuleList(
            [CrossAttentionBlock(dim, num_heads) for _ in range(depth)]
        )
        self.img_to_text_layers = nn.ModuleList(
            [CrossAttentionBlock(dim, num_heads) for _ in range(depth)]
        )

        self.gated_fusion = GatedFusion(dim)

        # 🔥 NEW: attention pooling for image tokens
        self.img_attn_pool = nn.Linear(dim, 1)

    def forward(self, text_tokens, img_tokens, text_pad_mask=None):
        """
        Args:
            text_tokens:   [B, T_text, dim]
            img_tokens:    [B, T_img, dim]
            text_pad_mask: [B, T_text] (True = PAD)
        """
        t = text_tokens
        v = img_tokens

        for t2i, i2t in zip(self.text_to_img_layers, self.img_to_text_layers):

            # Text attends to image
            t = t2i(
                t, v,
                key_padding_mask=None,             # image has no padding
                query_padding_mask=text_pad_mask   # mask text PAD queries
            )

            # Image attends to text
            v = i2t(
                v, t,
                key_padding_mask=text_pad_mask,    # mask text PAD keys
                query_padding_mask=None            # image has no padding
            )

        # 🔥 TEXT: CLS token
        text_pooled = t[:, 0]   # [B, dim]

        # 🔥 IMAGE: Attention pooling (instead of CLS)
        attn_weights = torch.softmax(self.img_attn_pool(v), dim=1)   # [B, T_img, 1]
        img_pooled   = (attn_weights * v).sum(dim=1)                 # [B, dim]

        return self.gated_fusion(text_pooled, img_pooled)


class VQAModel(nn.Module):
    def __init__(self, num_answers):
        super().__init__()

        self.vision_encoder = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True,
            num_classes=0
        )
        vit_dim = 768

        self.text_encoder = AutoModel.from_pretrained("bert-base-uncased")
        bert_dim = 768

        self.dim = 768

        self.vision_proj = nn.Sequential(
            nn.Linear(vit_dim, self.dim),
            nn.LayerNorm(self.dim),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(bert_dim, self.dim),
            nn.LayerNorm(self.dim),
        )

        self.co_attn = MultiLayerCoAttention(self.dim, depth=4, num_heads=8)

        self.classifier = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(self.dim, num_answers),
        )

        self._init_weights()

    def _init_weights(self):
        for module in [self.vision_proj, self.text_proj, self.classifier,
                       self.co_attn]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, image, input_ids, attention_mask):
        # Vision branch
        img_feats  = self.vision_encoder.forward_features(image)   # [B, 197, 768]
        img_tokens = self.vision_proj(img_feats)                    # [B, 197, 768]

        # Text branch
        text_out    = self.text_encoder(input_ids=input_ids,
                                        attention_mask=attention_mask)
        text_tokens = self.text_proj(text_out.last_hidden_state)    # [B, 32, 768]

        # ✅ Padding mask: True where token is [PAD]
        text_pad_mask = (attention_mask == 0)   # [B, 32]

        # Multi-layer co-attention + gated fusion
        fused = self.co_attn(text_tokens, img_tokens, text_pad_mask)  # [B, 768]

        return self.classifier(fused)   # [B, num_answers]


# =========================
# LOSS
# =========================
class VQASoftLoss(nn.Module):
    def __init__(self, smoothing=0.0, num_answers=3129):
        super().__init__()
        self.smoothing   = smoothing
        self.num_answers = num_answers

    def forward(self, logits, soft_labels):
        soft_labels = soft_labels.to(logits.device)
        if self.smoothing > 0:
            # Blend toward uniform: keeps soft label structure but smooths peaks
            smooth_floor = self.smoothing / self.num_answers
            soft_labels  = (1 - self.smoothing) * soft_labels + smooth_floor
        return F.binary_cross_entropy_with_logits(
            logits, soft_labels, reduction="mean"
        )

# =========================
# TRAIN SETUP
# =========================
device = CFG.DEVICE

# ✅ Start fresh — do NOT load the old checkpoint.
# It was trained entirely on incorrect one-hot labels, so its weights
# have fit the wrong distribution. Fresh init + correct soft labels
# will outperform fine-tuning from that checkpoint.
model = VQAModel(CFG.NUM_ANSWERS).to(device)
print("Training from scratch (fresh init with correct soft labels).")

# Differential learning rates
optimizer = torch.optim.AdamW([
    {"params": model.vision_encoder.parameters(), "lr": CFG.LR * 0.05},
    {"params": model.text_encoder.parameters(),   "lr": CFG.LR * 0.05},
    {"params": model.vision_proj.parameters(),    "lr": CFG.LR},
    {"params": model.text_proj.parameters(),      "lr": CFG.LR},
    {"params": model.co_attn.parameters(),        "lr": CFG.LR},
    {"params": model.classifier.parameters(),     "lr": CFG.LR},
], weight_decay=1e-2, eps=1e-8)

criterion = VQASoftLoss(smoothing=CFG.LABEL_SMOOTHING, num_answers=CFG.NUM_ANSWERS)

total_steps  = CFG.EPOCHS * len(train_loader) // CFG.ACCUMULATE
warmup_steps = CFG.WARMUP_EPOCHS * len(train_loader) // CFG.ACCUMULATE

def lr_lambda(step):
    if step < warmup_steps:
        return step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
scaler = torch.cuda.amp.GradScaler() if CFG.USE_AMP else None

# =========================
# TRAIN LOOP
# =========================
def train_one_epoch():
    model.train()
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    for i, batch in enumerate(train_loader):
        images    = batch["image"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attn_mask = batch["attention_mask"].to(device, non_blocking=True)
        soft      = batch["soft_label"].to(device, non_blocking=True)

        if CFG.USE_AMP:
            with autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(images, input_ids, attn_mask)
                loss    = criterion(outputs, soft) / CFG.ACCUMULATE
            scaler.scale(loss).backward()
        else:
            outputs = model(images, input_ids, attn_mask)
            loss    = criterion(outputs, soft) / CFG.ACCUMULATE
            loss.backward()

        if torch.isnan(loss):
            optimizer.zero_grad(set_to_none=True)
            continue

        is_last_batch = (i + 1 == len(train_loader))
        if (i + 1) % CFG.ACCUMULATE == 0 or is_last_batch:
            if CFG.USE_AMP:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.GRAD_CLIP)
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        total_loss += loss.item() * CFG.ACCUMULATE

    return total_loss / len(train_loader)


@torch.no_grad()
def evaluate():
    model.eval()
    correct, total = 0, 0

    for batch in val_loader:
        images    = batch["image"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attn_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels    = batch["label"].to(device, non_blocking=True)
        soft      = batch["soft_label"].to(device, non_blocking=True)

        if CFG.USE_AMP:
            with autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(images, input_ids, attn_mask)
        else:
            outputs = model(images, input_ids, attn_mask)

        preds     = outputs.argmax(1)
        batch_idx = torch.arange(len(preds), device=device)
        scores    = soft[batch_idx, preds]
        correct  += scores.sum().item()
        total    += labels.size(0)

    return correct / total


# =========================
# TRAIN
# =========================
best_acc = 0.0
logger   = EpochLogger(CFG.LOG_FILE)

try:
    for epoch in tqdm.tqdm(range(CFG.EPOCHS)):
        epoch_start = time.time()

        loss = train_one_epoch()
        acc  = evaluate()

        epoch_secs = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "best_vqa_model3.pt")

        logger.log(
            epoch      = epoch + 1,
            epoch_secs = epoch_secs,
            loss       = loss,
            val_acc    = acc,
            lr         = current_lr,
            best_acc   = best_acc,
        )

        print(f"\nEpoch {epoch+1}/{CFG.EPOCHS}")
        print(f"  Loss      : {loss:.4f}")
        print(f"  Val Acc   : {acc:.4f}  (VQA soft score)")
        print(f"  LR        : {current_lr:.2e}")
        print(f"  Epoch time: {epoch_secs:.1f}s  ({epoch_secs/60:.1f} min)")
        if acc >= best_acc:
            print(f"  ✓ Saved best model (acc={best_acc:.4f})")

finally:
    logger.close()

print(f"\nTraining complete. Best Val Acc: {best_acc:.4f}")
print(f"Full log saved to: {CFG.LOG_FILE}")