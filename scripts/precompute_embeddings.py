"""
precompute_embeddings.py

Runs CLIP once on all images and questions, saves embeddings to disk.
We never run CLIP during training — only load cached tensors.
"""

import json
import sys
from pathlib import Path
from tqdm import tqdm

import torch
import clip
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import get_device, get_logger

logger = get_logger("precompute")


def load_vqa_samples(questions_path, annotations_path):
    """Load questions and annotations, filter to yes/no only.
    
    WHY filter to yes/no?
    Keeps the task binary and the comparison between fusion
    strategies clean. We're studying fusion, not output heads.
    """
    with open(questions_path) as f:
        questions = {q["question_id"]: q
                     for q in json.load(f)["questions"]}

    with open(annotations_path) as f:
        annotations = json.load(f)["annotations"]

    samples = []
    for ann in annotations:
        if ann["answer_type"] != "yes/no":
            continue

        majority = ann["multiple_choice_answer"]
        if majority not in ("yes", "no"):
            continue

        qid = ann["question_id"]
        samples.append({
            "question_id": qid,
            "image_id":    ann["image_id"],
            "question":    questions[qid]["question"],
            "label":       1 if majority == "yes" else 0,
        })

    logger.info(f"Found {len(samples)} yes/no samples")
    return samples


def get_image_path(image_id, images_dir):
    """COCO images follow a fixed naming convention."""
    filename = f"COCO_val2014_{image_id:012d}.jpg"
    return Path(images_dir) / filename


def _save_chunk(lists, chunk_idx, tmp_dir):
    """Save accumulated tensors to a chunk file and clear the lists."""
    chunk_path = tmp_dir / f"chunk_{chunk_idx:04d}.pt"
    torch.save({
        "image_global":         torch.stack(lists["image_global"]),
        "image_patches":        torch.stack(lists["image_patches"]),
        "image_patches_layer4": torch.stack(lists["image_patches_l4"]),
        "image_patches_layer8": torch.stack(lists["image_patches_l8"]),
        "text_feat":            torch.stack(lists["text_feat"]),
        "text_tokens":          torch.stack(lists["text_tokens"]),
        "text_mask":            torch.stack(lists["text_mask"]),
        "labels":               torch.tensor(lists["labels"]),
    }, chunk_path)
    for v in lists.values():
        v.clear()
    return chunk_path


def precompute(config_path="configs/default.yaml"):
    import yaml
    import tempfile
    import gc

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = get_device()
    logger.info(f"Device: {device}")

    # load CLIP — we need the raw model to extract patch embeddings
    logger.info("Loading CLIP ViT-B/32...")
    model, preprocess = clip.load("ViT-B/32", device="cpu")
    model.eval()

    samples = load_vqa_samples(
        cfg["data"]["questions_path"],
        cfg["data"]["annotations_path"],
    )

    # ── Chunked accumulation to avoid OOM ─────────────────────────────
    CHUNK_SIZE = 10000
    lists = {
        "image_global": [], "image_patches": [],
        "image_patches_l4": [], "image_patches_l8": [],
        "text_feat": [], "text_tokens": [], "text_mask": [],
        "labels": [],
    }
    skipped    = 0
    chunk_idx  = 0
    out_path   = Path(cfg["data"]["embeddings_path"])
    tmp_dir    = out_path.parent / "_chunks_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    chunk_paths = []

    # layers to capture (0-indexed): 4th, 8th, 12th transformer block
    capture_layers = {3, 7, 11}

    with torch.no_grad():
        for sample in tqdm(samples, desc="Embedding"):
            img_path = get_image_path(
                sample["image_id"],
                cfg["data"]["images_dir"]
            )

            if not img_path.exists():
                skipped += 1
                continue

            try:
                # ── Image embeddings (multi-layer) ─────────────────────
                img = preprocess(
                    Image.open(img_path).convert("RGB")
                ).unsqueeze(0).to("cpu")

                visual = model.visual

                # patch embedding + positional encoding
                # ViT-B/32: 224/32 = 7 → 7x7 = 49 patches
                x = visual.conv1(img)                       # (1, 768, 7, 7)
                x = x.reshape(x.shape[0], x.shape[1], -1)  # (1, 768, 49)
                x = x.permute(0, 2, 1)                     # (1, 49, 768)

                # prepend CLS token
                cls = visual.class_embedding.unsqueeze(0).unsqueeze(0)
                cls = cls.expand(x.shape[0], -1, -1)
                x = torch.cat([cls, x], dim=1)             # (1, 50, 768)

                x = x + visual.positional_embedding
                x = visual.ln_pre(x)

                # run through transformer layer-by-layer, capturing intermediates
                x = x.permute(1, 0, 2)                     # (50, 1, 768)
                layer_patches = {}
                for i, block in enumerate(visual.transformer.resblocks):
                    x = block(x)
                    if i in capture_layers:
                        p = x.permute(1, 0, 2)[:, 1:, :]   # (1, 49, 768)
                        layer_patches[i] = (p.float() @ visual.proj.float())  # (1, 49, 512)

                x = x.permute(1, 0, 2)                     # (1, 50, 768)

                # global embedding — CLS token + projection (final layer)
                global_feat = x[:, 0, :].float() @ visual.proj.float()  # (1, 512)

                lists["image_global"].append(global_feat.squeeze(0).cpu().float())
                lists["image_patches"].append(layer_patches[11].squeeze(0).cpu().float())
                lists["image_patches_l4"].append(layer_patches[3].squeeze(0).cpu().float())
                lists["image_patches_l8"].append(layer_patches[7].squeeze(0).cpu().float())

                # ── Text embeddings (token-level + pooled) ─────────────
                tokens = clip.tokenize(
                    [sample["question"]], truncate=True
                ).to("cpu")                                  # (1, 77)

                # manual forward through text transformer
                x_text = model.token_embedding(tokens).float()  # (1, 77, 512)
                x_text = x_text + model.positional_embedding.float()
                x_text = x_text.permute(1, 0, 2)            # (77, 1, 512)

                for block in model.transformer.resblocks:
                    x_text = block(x_text)

                x_text = x_text.permute(1, 0, 2)            # (1, 77, 512)
                x_text = model.ln_final(x_text).float()      # (1, 77, 512)

                # project all 77 positions
                text_all_tokens = x_text @ model.text_projection.float()  # (1, 77, 512)

                # pooled text: EOS token position (same as encode_text)
                eos_idx = tokens.argmax(dim=-1)              # (1,)
                text_pooled = text_all_tokens[0, eos_idx[0], :]  # (512,)

                # padding mask: True for real tokens [0..eos_idx] inclusive
                mask = torch.zeros(77, dtype=torch.bool)
                mask[:eos_idx[0] + 1] = True

                lists["text_feat"].append(text_pooled.cpu().float())
                lists["text_tokens"].append(text_all_tokens.squeeze(0).cpu().float())
                lists["text_mask"].append(mask.cpu())

                lists["labels"].append(sample["label"])

                # ── Flush chunk to disk when full ─────────────────────
                if len(lists["labels"]) >= CHUNK_SIZE:
                    chunk_paths.append(_save_chunk(lists, chunk_idx, tmp_dir))
                    logger.info(f"Saved chunk {chunk_idx} ({CHUNK_SIZE} samples)")
                    chunk_idx += 1
                    gc.collect()

            except Exception as e:
                logger.warning(f"Failed qid={sample['question_id']}: {e}")
                skipped += 1

    # save remaining samples as final chunk
    if lists["labels"]:
        chunk_paths.append(_save_chunk(lists, chunk_idx, tmp_dir))
        logger.info(f"Saved chunk {chunk_idx} ({len(lists['labels'])} samples)")

    if skipped:
        logger.warning(f"Skipped {skipped} samples")

    # ── Concatenate all chunks into final file ────────────────────────
    logger.info(f"Merging {len(chunk_paths)} chunks...")
    all_data = {}
    for cp in chunk_paths:
        chunk = torch.load(cp, weights_only=True)
        for key, val in chunk.items():
            all_data.setdefault(key, []).append(val)
        del chunk
        gc.collect()

    merged = {key: torch.cat(vals, dim=0) for key, vals in all_data.items()}
    del all_data
    gc.collect()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, str(out_path))

    # clean up temp chunks
    for cp in chunk_paths:
        cp.unlink()
    tmp_dir.rmdir()

    n = len(merged["labels"])
    yes = int(merged["labels"].sum().item())
    logger.info(f"Saved {n} samples → {out_path}")
    logger.info(f"Yes: {yes} ({yes/n*100:.1f}%)  No: {n-yes} ({(n-yes)/n*100:.1f}%)")


if __name__ == "__main__":
    precompute()