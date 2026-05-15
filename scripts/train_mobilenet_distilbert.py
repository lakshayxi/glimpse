# =========================
# IMPORTS
# =========================
import os
import json
import pandas as pd
from PIL import Image
from collections import Counter
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from transformers import AutoTokenizer, AutoModel
import timm

from torch.amp import autocast, GradScaler

torch.backends.cudnn.benchmark = True

# =========================
# CONFIG
# =========================
class CFG:
    IMG_SIZE = 224
    MAX_LEN = 32
    BATCH_SIZE = 64
    EPOCHS = 50
    LR = 3e-4
    NUM_ANSWERS = 1000
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    PATIENCE = 5
    LOG_FILE = "training_log0.csv"

# =========================
# PATHS (COCO 2014)
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
def load_vqa(q_path, a_path, split):
    with open(q_path) as f:
        questions = json.load(f)["questions"]
    with open(a_path) as f:
        annotations = json.load(f)["annotations"]

    q_df = pd.DataFrame(questions)
    a_df = pd.DataFrame(annotations)

    df = q_df.merge(a_df, on="question_id", suffixes=("_q", "_a"))

    img_col = "image_id_q" if "image_id_q" in df.columns else "image_id_a"

    prefix = f"COCO_{split}2014_"
    df["image"] = df[img_col].apply(
        lambda x: f"{prefix}{str(x).zfill(12)}.jpg"
    )

    df["answer"] = df["multiple_choice_answer"]

    return df[["image", "question", "answer"]]

train_df = load_vqa(TRAIN_Q, TRAIN_A, "train")
val_df   = load_vqa(VAL_Q, VAL_A, "val")

# =========================
# FILTER IMAGES
# =========================
def filter_existing(df, img_dir):
    mask = df["image"].apply(lambda x: os.path.exists(os.path.join(img_dir, x)))
    print(f"[{img_dir}] {mask.sum()} / {len(df)} images found")
    return df[mask].reset_index(drop=True)

train_df = filter_existing(train_df, TRAIN_IMG_DIR)
val_df   = filter_existing(val_df, VAL_IMG_DIR)

# small subset for speed
train_df = train_df.sample(min(10000, len(train_df))).reset_index(drop=True)
val_df   = val_df.sample(min(2000, len(val_df))).reset_index(drop=True)

# =========================
# ANSWER VOCAB
# =========================
counter = Counter(train_df["answer"])
most_common = counter.most_common(CFG.NUM_ANSWERS)
answer_to_idx = {ans: idx for idx, (ans, _) in enumerate(most_common)}

# =========================
# TOKENIZER
# =========================
os.environ["TOKENIZERS_PARALLELISM"] = "false"
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# =========================
# TRANSFORMS
# =========================
transform = transforms.Compose([
    transforms.Resize((CFG.IMG_SIZE, CFG.IMG_SIZE)),
    transforms.ToTensor(),
])

# =========================
# DATASET
# =========================
class VQADataset(Dataset):
    def __init__(self, df, img_dir):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir

        enc = tokenizer(
            self.df["question"].tolist(),
            padding="max_length",
            truncation=True,
            max_length=CFG.MAX_LEN,
            return_tensors="pt"
        )

        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = os.path.join(self.img_dir, row["image"])
        image = Image.open(img_path).convert("RGB")
        image = transform(image)

        label = answer_to_idx.get(row["answer"], 0)

        return {
            "image": image,
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "label": torch.tensor(label)
        }

# =========================
# DATALOADER
# =========================
train_loader = DataLoader(
    VQADataset(train_df, TRAIN_IMG_DIR),
    batch_size=CFG.BATCH_SIZE,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True
)

val_loader = DataLoader(
    VQADataset(val_df, VAL_IMG_DIR),
    batch_size=CFG.BATCH_SIZE,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True
)

# =========================
# MODEL
# =========================
class CrossAttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, 8, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.ReLU(),
            nn.Linear(dim*4, dim)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, q, kv):
        attn_out, _ = self.attn(q, kv, kv)
        x = self.norm1(q + attn_out)
        return self.norm2(x + self.ff(x))

class VQAModel(nn.Module):
    def __init__(self, num_answers):
        super().__init__()

        self.vision_encoder = timm.create_model(
            "mobilenetv3_small_100",
            pretrained=True,
            num_classes=0
        )

        self.text_encoder = AutoModel.from_pretrained("distilbert-base-uncased")

        self.dim = 512

        self.vision_proj = nn.Linear(1024, self.dim)
        self.text_proj = nn.Linear(768, self.dim)

        self.cross_attn = CrossAttentionBlock(self.dim)

        self.classifier = nn.Sequential(
            nn.Linear(self.dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_answers)
        )

    def forward(self, image, input_ids, attention_mask):
        img_feat = self.vision_encoder(image)
        img_tokens = self.vision_proj(img_feat.unsqueeze(1))

        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_tokens = self.text_proj(text_out.last_hidden_state)

        fused = self.cross_attn(text_tokens, img_tokens)

        return self.classifier(fused[:, 0])

# =========================
# TRAIN SETUP
# =========================
device = CFG.DEVICE
model = VQAModel(CFG.NUM_ANSWERS).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.LR)
criterion = nn.CrossEntropyLoss()
scaler = GradScaler("cuda")

# =========================
# TRAIN LOOP
# =========================
def train_one_epoch(epoch):
    model.train()
    total_loss = 0

    pbar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}", ncols=100)

    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        with autocast("cuda"):
            outputs = model(images, input_ids, attention_mask)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    return total_loss / len(train_loader)

def evaluate():
    model.eval()
    correct, total = 0, 0

    pbar = tqdm(val_loader, desc="Validation", ncols=100)

    with torch.no_grad():
        for batch in pbar:
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images, input_ids, attention_mask)
            preds = outputs.argmax(1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix(acc=correct / total)

    return correct / total

# =========================
# CSV INIT
# =========================
if not os.path.exists(CFG.LOG_FILE):
    pd.DataFrame(columns=["epoch", "loss", "val_acc"]).to_csv(CFG.LOG_FILE, index=False)

# =========================
# TRAINING + EARLY STOPPING
# =========================
best_acc = 0
patience_counter = 0

for epoch in range(CFG.EPOCHS):
    loss = train_one_epoch(epoch)
    acc = evaluate()

    print(f"\nEpoch {epoch+1}")
    print(f"Loss: {loss:.4f}")
    print(f"Val Acc: {acc:.4f}")

    # save log
    pd.DataFrame([{
        "epoch": epoch+1,
        "loss": loss,
        "val_acc": acc
    }]).to_csv(CFG.LOG_FILE, mode="a", header=False, index=False)

    # early stopping
    if acc > best_acc:
        best_acc = acc
        patience_counter = 0
        torch.save(model.state_dict(), "best_model.pth")
        print("✅ Model improved & saved")
    else:
        patience_counter += 1
        print(f"⚠️ No improvement ({patience_counter}/{CFG.PATIENCE})")

        if patience_counter >= CFG.PATIENCE:
            print("🛑 Early stopping triggered")
            break