"""
run_lora_training.py

Trains the best performing fusion model (bilinear) with LoRA-adapted CLIP.
We only run LoRA on bilinear since it was the strongest model in the
frozen experiment — gives the cleanest comparison:

  bilinear + frozen CLIP  vs  bilinear + LoRA CLIP
"""

import yaml
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import BilinearFusion
from src.train_lora import train_lora


def main():
    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)

    # we only run LoRA on bilinear — our best model
    # WHY only bilinear? Clean controlled comparison:
    # same fusion architecture, only CLIP adaptation changes
    model = BilinearFusion(
        embed_dim=config["model"]["embed_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        num_classes=config["model"]["num_classes"],
        dropout=config["model"]["dropout"],
    )

    print("\n" + "="*50)
    print("Training: bilinear + LoRA CLIP")
    print("="*50)

    train_lora(model, config, fusion_name="bilinear")


if __name__ == "__main__":
    main()