"""
dataset_finetune.py

Dataset for LoRA fine-tuning. Unlike VQADataset which loads
cached embeddings, this loads raw images and questions so
CLIP can process them live during training — allowing gradients
to flow back through CLIP's LoRA layers.

WHY not use cached embeddings for LoRA?
  Cached embeddings are fixed tensors computed by frozen CLIP.
  If we want CLIP's LoRA layers to learn, we must run CLIP
  forward on every batch during training so gradients can
  flow back through the LoRA matrices.
"""

import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


class VQAFinetuneDataset(Dataset):
    def __init__(self, questions_path: str, annotations_path: str,
                 images_dir: str, preprocess, answer_type: str = "yes/no"):
        """
        Args:
            questions_path:   path to VQA questions JSON
            annotations_path: path to VQA annotations JSON
            images_dir:       directory containing COCO images
            preprocess:       CLIP's image preprocessing transform
            answer_type:      filter to this answer type only
        """
        self.images_dir = Path(images_dir)
        self.preprocess = preprocess

        # load and filter samples
        self.samples = self._load_samples(
            questions_path, annotations_path, answer_type
        )

    def _load_samples(self, questions_path, annotations_path, answer_type):
        with open(questions_path) as f:
            questions = {q["question_id"]: q["question"]
                        for q in json.load(f)["questions"]}

        with open(annotations_path) as f:
            annotations = json.load(f)["annotations"]

        samples = []
        for ann in annotations:
            if ann["answer_type"] != answer_type:
                continue
            majority = ann["multiple_choice_answer"]
            if majority not in ("yes", "no"):
                continue

            img_path = self._get_image_path(ann["image_id"])
            if not img_path.exists():
                continue

            samples.append({
                "image_path": str(img_path),
                "question":   questions[ann["question_id"]],
                "label":      1 if majority == "yes" else 0,
            })

        samples = samples[:5000]  # smoke test — remove this line for full run
        print(f"Loaded {len(samples)} yes/no samples")
        return samples

    def _get_image_path(self, image_id: int) -> Path:
        filename = f"COCO_val2014_{image_id:012d}.jpg"
        return self.images_dir / filename

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # load and preprocess image
        # WHY convert RGB? some COCO images are grayscale — CLIP expects 3 channels
        image = self.preprocess(
            Image.open(sample["image_path"]).convert("RGB")
        )

        return {
            "image":    image,                          # (3, 224, 224)
            "question": sample["question"],             # raw string
            "label":    torch.tensor(sample["label"]), # 0 or 1
        }