"""
run_training.py

Trains all three fusion models one by one.
Run this after precompute_embeddings.py has finished.
"""

import yaml
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import ConcatMLP, BilinearFusion, CrossAttentionFusion, CrossAttentionFusionV2
from src.train import train


def main():
    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)

    embed_dim  = config["model"]["embed_dim"]
    hidden_dim = config["model"]["hidden_dim"]
    num_classes = config["model"]["num_classes"]
    dropout    = config["model"]["dropout"]
    num_heads  = config["model"]["num_heads"]

    models = {
        # "concat_mlp": ConcatMLP(
        #     embed_dim=embed_dim,
        #     hidden_dim=hidden_dim,
        #     num_classes=num_classes,
        #     dropout=dropout,
        # ),
        # "bilinear": BilinearFusion(
        #     embed_dim=embed_dim,
        #     hidden_dim=hidden_dim,
        #     num_classes=num_classes,
        #     dropout=dropout,
        # ),
        # "cross_attention": CrossAttentionFusion(
        #     embed_dim=embed_dim,
        #     num_heads=num_heads,
        #     dropout=dropout,
        #     num_classes=num_classes,
        # ),
        "cross_attention_v2": CrossAttentionFusionV2(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=config["model"]["dropout"],
            num_classes=num_classes,
            num_layers=config["model"]["num_layers"],
        ),
    }
    

    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training: {name}")
        print(f"{'='*50}")
        train(model, config, fusion_name=name)


if __name__ == "__main__":
    main()