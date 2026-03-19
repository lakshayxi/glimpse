import torch
from torch.utils.data import Dataset

class VQADataset(Dataset):
    def __init__(self, embeddings_path: str):

        data = torch.load(embeddings_path, map_location="cpu")

        self.image_global= data["image_global"] 
        self.image_patches = data["image_patches"]
        self.text_feat = data["text_feat"]
        self.labels = data["labels"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "image_global": self.image_global[idx],
            "image_patches": self.image_patches[idx],
            "text_feat": self.text_feat[idx],
            "label": self.labels[idx],
        }
