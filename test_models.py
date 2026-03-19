import torch
from src.models import ConcatMLP, BilinearFusion, CrossAttentionFusion

batch_size = 4

# global features — used by ConcatMLP and BilinearFusion
image_feat = torch.randn(batch_size, 512)
text_feat  = torch.randn(batch_size, 512)

# patch-level features — used by CrossAttentionFusion
# 196 patches, each 512d
image_patches = torch.randn(batch_size, 196, 512)

print("Testing ConcatMLP...")
model = ConcatMLP()
out = model(image_feat, text_feat)
print(f"  input: image (4,512) + text (4,512) → output {tuple(out.shape)}")

print("Testing BilinearFusion...")
model = BilinearFusion()
out = model(image_feat, text_feat)
print(f"  input: image (4,512) + text (4,512) → output {tuple(out.shape)}")

print("Testing CrossAttentionFusion...")
model = CrossAttentionFusion()
out = model(image_patches, text_feat)
print(f"  input: image (4,196,512) + text (4,512) → output {tuple(out.shape)}")


print("Testing CrossAttentionFusionV2...")
from src.models import CrossAttentionFusionV2
model = CrossAttentionFusionV2()
out = model(image_patches, text_feat)
print(f"  input: image (4,196,512) + text (4,512) → output {tuple(out.shape)}")