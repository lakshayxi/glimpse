import torch
from src.models import (
    ConcatMLP, BilinearFusion, CrossAttentionFusion, CrossAttentionFusionV2,
    GeometryFusion, TokenGrounding, LayerAdaptiveFusion, MultiGlimpse,
    pool_patches,
)

batch_size = 4

# ── Shared test tensors ──────────────────────────────────────────────

# global features — used by ConcatMLP, BilinearFusion, GeometryFusion
image_feat = torch.randn(batch_size, 512)
text_feat  = torch.randn(batch_size, 512)

# patch-level features — used by CrossAttention, LayerAdaptive, etc.
image_patches = torch.randn(batch_size, 49, 512)

# token-level text — used by TokenGrounding, MultiGlimpse
text_tokens = torch.randn(batch_size, 77, 512)
text_mask = torch.zeros(batch_size, 77, dtype=torch.bool)
text_mask[:, :10] = True  # simulate 10-token questions

# multi-layer patches — used by LayerAdaptiveFusion
patches_l4 = torch.randn(batch_size, 49, 512)
patches_l8 = torch.randn(batch_size, 49, 512)


# ── Original models ──────────────────────────────────────────────────

print("Testing ConcatMLP...")
model = ConcatMLP()
out = model(image_feat, text_feat)
assert out.shape == (batch_size, 2), f"Expected ({batch_size}, 2), got {out.shape}"
print(f"  input: image ({batch_size},512) + text ({batch_size},512) → output {tuple(out.shape)}")

print("Testing BilinearFusion...")
model = BilinearFusion()
out = model(image_feat, text_feat)
assert out.shape == (batch_size, 2), f"Expected ({batch_size}, 2), got {out.shape}"
print(f"  input: image ({batch_size},512) + text ({batch_size},512) → output {tuple(out.shape)}")

print("Testing CrossAttentionFusion...")
model = CrossAttentionFusion()
out = model(image_patches, text_feat)
assert out.shape == (batch_size, 2), f"Expected ({batch_size}, 2), got {out.shape}"
print(f"  input: image ({batch_size},49,512) + text ({batch_size},512) → output {tuple(out.shape)}")

print("Testing CrossAttentionFusionV2...")
model = CrossAttentionFusionV2()
out = model(image_patches, text_feat)
assert out.shape == (batch_size, 2), f"Expected ({batch_size}, 2), got {out.shape}"
print(f"  input: image ({batch_size},49,512) + text ({batch_size},512) → output {tuple(out.shape)}")


# ── Novel architectures ──────────────────────────────────────────────

print("\nTesting GeometryFusion...")
model = GeometryFusion()
out = model(image_feat, text_feat)
assert out.shape == (batch_size, 2), f"Expected ({batch_size}, 2), got {out.shape}"
print(f"  input: image ({batch_size},512) + text ({batch_size},512) → output {tuple(out.shape)}")

print("Testing TokenGrounding...")
model = TokenGrounding()
out = model(image_patches, text_tokens, text_mask)
assert out.shape == (batch_size, 2), f"Expected ({batch_size}, 2), got {out.shape}"
print(f"  input: patches ({batch_size},49,512) + tokens ({batch_size},77,512) + mask ({batch_size},77) → output {tuple(out.shape)}")

print("Testing LayerAdaptiveFusion...")
model = LayerAdaptiveFusion()
out = model(image_patches, patches_l4, patches_l8, text_feat)
assert out.shape == (batch_size, 2), f"Expected ({batch_size}, 2), got {out.shape}"
print(f"  input: patches_12+4+8 ({batch_size},49,512)x3 + text ({batch_size},512) → output {tuple(out.shape)}")

print("Testing MultiGlimpse...")
model = MultiGlimpse()
out = model(image_patches, text_tokens, text_mask)
assert out.shape == (batch_size, 2), f"Expected ({batch_size}, 2), got {out.shape}"
print(f"  input: patches ({batch_size},49,512) + tokens ({batch_size},77,512) + mask ({batch_size},77) → output {tuple(out.shape)}")


# ── Utility tests ────────────────────────────────────────────────────

print("\nTesting pool_patches utility...")
p4 = pool_patches(image_patches, 2, 2)
assert p4.shape == (batch_size, 4, 512), f"Expected ({batch_size},4,512), got {p4.shape}"
p16 = pool_patches(image_patches, 4, 4)
assert p16.shape == (batch_size, 16, 512), f"Expected ({batch_size},16,512), got {p16.shape}"
print(f"  pool_patches: (4,49,512) → (4,4,512) ✓, (4,16,512) ✓")


# ── Gradient flow test ───────────────────────────────────────────────

print("\nTesting gradient flow through MultiGlimpse...")
model = MultiGlimpse()
img = torch.randn(2, 49, 512, requires_grad=True)
txt = torch.randn(2, 77, 512, requires_grad=True)
msk = torch.ones(2, 77, dtype=torch.bool)
out = model(img, txt, msk)
out.sum().backward()
assert img.grad is not None, "No gradient on image_patches"
assert txt.grad is not None, "No gradient on text_tokens"
print(f"  gradients flow to image_patches ✓ and text_tokens ✓")


print("\n" + "="*50)
print("All tests passed!")
print("="*50)
