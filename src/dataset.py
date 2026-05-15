import torch
from pathlib import Path
from torch.utils.data import Dataset

_ALL_KEYS = (
    "image_global", "image_patches", "text_feat",
    "text_tokens", "text_mask",
    "image_patches_layer4", "image_patches_layer8",
)


class VQADataset(Dataset):
    """Loads precomputed CLIP embeddings for VQA.

    Supports two storage formats:
      1. Single .pt file (legacy): all keys in one dict
      2. Per-key directory: data/embeddings/keys/{key}.pt (memory-efficient)

    Only loads the keys the model actually needs (via `required_keys`),
    keeping RAM usage manageable on 16 GB machines.
    """

    def __init__(self, embeddings_path: str, required_keys=None):
        embeddings_path = Path(embeddings_path)
        keys_dir = embeddings_path.parent / "keys"

        # Determine which keys to load
        if required_keys is None:
            # Default: load everything available (backward compat)
            keys_to_load = list(_ALL_KEYS)
        else:
            keys_to_load = list(required_keys)

        # Always need labels
        self._loaded_keys = []

        if keys_dir.exists() and (keys_dir / "labels.pt").exists():
            # ── Per-key directory format ─────────────────────────
            self.labels = torch.load(
                str(keys_dir / "labels.pt"), map_location="cpu", weights_only=True
            )
            for key in keys_to_load:
                kp = keys_dir / f"{key}.pt"
                if kp.exists():
                    tensor = torch.load(str(kp), map_location="cpu", weights_only=True)
                    # Convert float16 back to float32 for training
                    if tensor.dtype == torch.float16:
                        tensor = tensor.float()
                    setattr(self, key, tensor)
                    self._loaded_keys.append(key)
        elif embeddings_path.exists():
            # ── Legacy single-file format ────────────────────────
            data = torch.load(str(embeddings_path), map_location="cpu", weights_only=True)
            self.labels = data["labels"]
            for key in keys_to_load:
                if key in data:
                    setattr(self, key, data[key])
                    self._loaded_keys.append(key)
        else:
            raise FileNotFoundError(
                f"No embeddings found at {embeddings_path} or {keys_dir}/"
            )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {"label": self.labels[idx]}
        for key in self._loaded_keys:
            item[key] = getattr(self, key)[idx]
        return item
