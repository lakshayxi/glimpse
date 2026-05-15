"""
merge_chunks.py — Memory-efficient merge of embedding chunks.

Saves each key as a separate .pt file in float16 (where applicable).
The dataset loads only the keys each model needs, keeping RAM usage
well under 16 GB even for the most demanding model (~12 GB for LayerAdaptive).
"""

import gc
import torch
from pathlib import Path

CHUNKS_DIR = Path("data/embeddings/_chunks_tmp")
OUT_DIR = Path("data/embeddings/keys")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Step 1: count total samples ──────────────────────────────────────
chunk_paths = sorted(CHUNKS_DIR.glob("chunk_*.pt"))
print(f"Found {len(chunk_paths)} chunks")

total_n = 0
for cp in chunk_paths:
    chunk = torch.load(cp, weights_only=True)
    n = len(chunk["labels"])
    print(f"  {cp.name}: {n} samples")
    total_n += n
    del chunk
    gc.collect()

print(f"Total: {total_n} samples\n")

# ── Step 2: merge each key independently ─────────────────────────────
KEYS = [
    "image_global", "image_patches",
    "image_patches_layer4", "image_patches_layer8",
    "text_feat", "text_tokens", "text_mask", "labels",
]

# Keys that benefit from float16
FLOAT16_KEYS = {"image_global", "image_patches", "image_patches_layer4",
                "image_patches_layer8", "text_feat", "text_tokens"}

for key in KEYS:
    out_path = OUT_DIR / f"{key}.pt"
    if out_path.exists():
        print(f"'{key}' already merged, skipping")
        continue

    print(f"Merging '{key}'...")
    parts = []
    for cp in chunk_paths:
        chunk = torch.load(cp, weights_only=True)
        parts.append(chunk[key])
        del chunk
        gc.collect()

    merged = torch.cat(parts, dim=0)
    del parts
    gc.collect()

    if key in FLOAT16_KEYS:
        merged = merged.half()

    print(f"  → {tuple(merged.shape)}, dtype={merged.dtype}")
    torch.save(merged, str(out_path))
    del merged
    gc.collect()

# ── Step 3: verify ───────────────────────────────────────────────────
print(f"\nVerifying saved keys in {OUT_DIR}/:")
for key in KEYS:
    kp = OUT_DIR / f"{key}.pt"
    t = torch.load(str(kp), weights_only=True)
    print(f"  {key}: {tuple(t.shape)}, dtype={t.dtype}")
    del t
    gc.collect()

# ── Step 4: cleanup chunks ──────────────────────────────────────────
print("\nCleaning up chunks...")
for cp in chunk_paths:
    cp.unlink()
CHUNKS_DIR.rmdir()
print("Done! Dataset should now load from data/embeddings/keys/")
