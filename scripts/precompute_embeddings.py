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


def precompute(config_path="configs/default.yaml"):
    import yaml
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

    image_global_list  = []
    image_patches_list = []
    text_feat_list     = []
    labels             = []
    skipped            = 0

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
                # ── Image embeddings ─────────────────────────────────────
                img = preprocess(
                    Image.open(img_path).convert("RGB")
                ).unsqueeze(0).to("cpu")

                # we hook into CLIP's visual transformer to get patch tokens
                # before the final pooling step
                visual = model.visual

                # run image through patch embedding + positional encoding
                x = visual.conv1(img)                    # (1, 768, 14, 14)
                x = x.reshape(x.shape[0], x.shape[1], -1)  # (1, 768, 196)
                x = x.permute(0, 2, 1)                  # (1, 196, 768)

                # prepend CLS token
                cls = visual.class_embedding.unsqueeze(0).unsqueeze(0)
                cls = cls.expand(x.shape[0], -1, -1)
                x = torch.cat([cls, x], dim=1)          # (1, 197, 768)

                # add positional embeddings
                x = x + visual.positional_embedding

                # layer norm before transformer
                x = visual.ln_pre(x)

                # run through transformer
                x = x.permute(1, 0, 2)                  # (197, 1, 768)
                x = visual.transformer(x)
                x = x.permute(1, 0, 2)                  # (1, 197, 768)

                # patch tokens — skip CLS (index 0), take remaining 196
                patches = x[:, 1:, :]                   # (1, 196, 768)

                # project patches to 512d using CLIP's projection
                patches = patches @ visual.proj          # (1, 196, 512)

                # global embedding — CLS token + projection
                global_feat = x[:, 0, :] @ visual.proj  # (1, 512)

                image_global_list.append(
                    global_feat.squeeze(0).cpu().float()
                )
                image_patches_list.append(
                    patches.squeeze(0).cpu().float()
                )

                # ── Text embedding ───────────────────────────────────────
                tokens = clip.tokenize(
                    [sample["question"]], truncate=True
                ).to("cpu")
                text_feat = model.encode_text(tokens)    # (1, 512)
                text_feat_list.append(
                    text_feat.squeeze(0).cpu().float()
                )

                labels.append(sample["label"])

            except Exception as e:
                logger.warning(f"Failed qid={sample['question_id']}: {e}")
                skipped += 1

    if skipped:
        logger.warning(f"Skipped {skipped} samples")

    # save everything to a single .pt file
    out_path = cfg["data"]["embeddings_path"]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "image_global":   torch.stack(image_global_list),   # (N, 512)
        "image_patches":  torch.stack(image_patches_list),  # (N, 196, 512)
        "text_feat":      torch.stack(text_feat_list),      # (N, 512)
        "labels":         torch.tensor(labels),             # (N,)
    }, out_path)

    n = len(labels)
    yes = sum(labels)
    logger.info(f"Saved {n} samples → {out_path}")
    logger.info(f"Yes: {yes} ({yes/n*100:.1f}%)  No: {n-yes} ({(n-yes)/n*100:.1f}%)")


if __name__ == "__main__":
    precompute()