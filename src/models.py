import torch
import torch.nn as nn
import torch.nn.functional as F

class ConcatMLP(nn.Module):
    REQUIRED_INPUTS = ("image_global", "text_feat")

    def __init__(self, embed_dim=512, hidden_dim=1024, num_classes=2, dropout=0.3):
        super().__init__()

        #WHY Concatenate? simplest way to do so imo
        #img(512)+text(512) = 1024d input to MLP

        self.mlp=nn.Sequential(
            nn.Linear(embed_dim *2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, image_feat, text_feat):
        #concat along the feature dimmension
        x= torch.cat([image_feat, text_feat], dim=-1) #(B, 1024)
        return self.mlp(x) # (B, 2)    

class BilinearFusion(nn.Module):
    REQUIRED_INPUTS = ("image_global", "text_feat")

    def __init__(self, embed_dim=512, hidden_dim=1024, num_classes=2, dropout=0.3):
        super().__init__()

        # project both into same space before multiplying
        self.image_proj = nn.Linear(embed_dim, hidden_dim)
        self.text_proj  = nn.Linear(embed_dim, hidden_dim)

        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, image_feat, text_feat):
        # project both to hidden_dim
        img = self.image_proj(image_feat)   # (B, 1024)
        txt = self.text_proj(text_feat)     # (B, 1024)

        # element-wise multiply, this captures interactions
        fused = img * txt                   # (B, 1024)
        return self.classifier(fused)       # (B, 2)        

class CrossAttentionFusion(nn.Module):
    REQUIRED_INPUTS = ("image_patches", "text_feat")

    def __init__(self, embed_dim=512, num_heads=8, dropout=0.3, num_classes=2):
        super().__init__()

        # WHY num_heads=8?
        # Multi-head attention runs 8 independent attention operations in parallel.
        # Each head can learn to focus on different aspects —
        # one head might focus on objects, another on spatial relationships.
        # Their outputs are concatenated and projected back to embed_dim.
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # expects (batch, seq, features) not (seq, batch, features)
        )

        # WHY LayerNorm?
        # After attention, the values can vary wildly in scale.
        # LayerNorm stabilizes them — makes training faster and more stable.
        # This is the standard residual + norm block from the Transformer paper.
        self.norm = nn.LayerNorm(embed_dim)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )

    def forward(self, image_patches, text_feat):
        # image_patches: (B, 196, 512) — 196 patch vectors
        # text_feat:     (B, 512)      — one question vector

        # text needs a sequence dimension to act as query
        # (B, 512) → (B, 1, 512)
        txt = text_feat.unsqueeze(1)

        # cross attention:
        # query = text (what are we looking for?)
        # key   = image patches (what does each region contain?)
        # value = image patches (what to return from relevant regions?)
        attended, _ = self.cross_attn(
            query=txt,          # (B, 1, 512)
            key=image_patches,  # (B, 196, 512)
            value=image_patches # (B, 196, 512)
        )
        # attended: (B, 1, 512) — question-guided image representation

        # squeeze back to (B, 512)
        attended = attended.squeeze(1)

        # residual connection — add original text back in
        # WHY? so the model doesn't forget the question after attending
        out = self.norm(attended + text_feat)

        return self.classifier(out)  # (B, 2)     # (B, 2) 

class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, dropout=0.3):
        super().__init__()

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # FFN — two linear layers with ReLU
        # WHY 4x hidden dim? Standard transformer convention.
        # The FFN expands then contracts — gives more transformation capacity.
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )

        # layer norms — one after attention, one after FFN
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, image_patches):
        # query:         (B, 1, 512)
        # image_patches: (B, 196, 512)

        # cross attention + residual + norm
        attended, _ = self.cross_attn(
            query=query,
            key=image_patches,
            value=image_patches
        )
        query = self.norm1(query + self.dropout(attended))

        # feedforward + residual + norm
        query = self.norm2(query + self.dropout(self.ffn(query)))

        return query  # (B, 1, 512)

class CrossAttentionFusionV2(nn.Module):
    REQUIRED_INPUTS = ("image_patches", "text_feat")

    def __init__(self, embed_dim=512, num_heads=8,
                 num_layers=2, dropout=0.3, num_classes=2):
        super().__init__()

        # WHY a list of layers?
        # We stack num_layers cross attention blocks.
        # Each block = cross attention + feedforward network.
        # nn.ModuleList tells PyTorch to track all layers properly.
        self.layers = nn.ModuleList([
            CrossAttentionBlock(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # final classifier
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )

    def forward(self, image_patches, text_feat):
        # image_patches: (B, 196, 512)
        # text_feat:     (B, 512)

        # text is our query — it gets refined through each layer
        # starts as the question vector, gets richer after each block
        query = text_feat.unsqueeze(1)  # (B, 1, 512)

        for layer in self.layers:
            query = layer(query, image_patches)

        # squeeze back to (B, 512)
        out = query.squeeze(1)
        return self.classifier(out)


# ── Novel Architectures ───────────────────────────────────────────────


class GeometryFusion(nn.Module):
    """Contrastive Geometry Decomposition.

    Exploits CLIP's pretrained contrastive geometry by decomposing
    image-text relationships into three orthogonal signals:
      alignment  — cosine similarity (what CLIP already knows)
      residual   — img - txt (visual "surprisal" beyond the question)
      interaction — img * txt (co-occurring features)
    """
    REQUIRED_INPUTS = ("image_global", "text_feat")

    def __init__(self, embed_dim=512, hidden_dim=512, num_classes=2, dropout=0.3):
        super().__init__()
        # alignment(1) + residual(512) + interaction(512) = 1025
        input_dim = 2 * embed_dim + 1

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, image_feat, text_feat):
        # L2-normalize to respect CLIP's hyperspherical geometry
        img = F.normalize(image_feat, dim=-1)
        txt = F.normalize(text_feat, dim=-1)

        alignment   = (img * txt).sum(dim=-1, keepdim=True)  # (B, 1)
        residual    = img - txt                                # (B, 512)
        interaction = img * txt                                # (B, 512)

        combined = torch.cat([alignment, residual, interaction], dim=-1)  # (B, 1025)
        return self.classifier(combined)  # (B, 2)


class TokenGrounding(nn.Module):
    """Token-Level Cross-Modal Grounding.

    Uses per-token text embeddings as MULTIPLE cross-attention queries
    against image patches. Each word in the question independently
    attends to image regions, enabling compositional reasoning.

    "red" attends to red-colored patches, "car" attends to car-shaped
    patches. The intersection reveals if there's a red car.
    """
    REQUIRED_INPUTS = ("image_patches", "text_tokens", "text_mask")

    def __init__(self, embed_dim=512, num_heads=8, num_layers=2,
                 dropout=0.3, num_classes=2):
        super().__init__()

        # reuse the existing CrossAttentionBlock — it already supports
        # arbitrary-length query sequences via nn.MultiheadAttention
        self.layers = nn.ModuleList([
            CrossAttentionBlock(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.output_norm = nn.LayerNorm(embed_dim)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )

    def forward(self, image_patches, text_tokens, text_mask):
        # image_patches: (B, 196, 512)
        # text_tokens:   (B, 77, 512) — all 77 positions, padded
        # text_mask:     (B, 77)      — True for real tokens

        query = text_tokens  # (B, 77, 512) — multiple queries

        for layer in self.layers:
            query = layer(query, image_patches)  # (B, 77, 512)

        # masked mean pooling: average only real token positions
        mask_expanded = text_mask.unsqueeze(-1).float()  # (B, 77, 1)
        pooled = (query * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)

        pooled = self.output_norm(pooled)  # (B, 512)
        return self.classifier(pooled)     # (B, 2)


class LayerAdaptiveFusion(nn.Module):
    """Layer-Adaptive Feature Extraction.

    Extracts patch features from CLIP visual layers 4, 8, and 12.
    A text-conditioned gating network learns which layer matters
    per question type:
      "Is the sky blue?"        → weights early layers (color)
      "Is there a dog?"         → weights late layers  (semantics)
      "Is the person sitting?"  → weights mid layers   (pose/spatial)
    """
    REQUIRED_INPUTS = (
        "image_patches",          # layer 12: (B, 196, 512)
        "image_patches_layer4",   # layer 4:  (B, 196, 512)
        "image_patches_layer8",   # layer 8:  (B, 196, 512)
        "text_feat",              # pooled:   (B, 512)
    )

    def __init__(self, embed_dim=512, num_heads=8, num_layers=2,
                 dropout=0.3, num_classes=2, num_source_layers=3):
        super().__init__()

        # text-conditioned gating: text_feat → softmax weights for 3 layers
        self.gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, num_source_layers),
        )

        # per-layer LayerNorm to handle scale differences between
        # intermediate and final layer features
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(num_source_layers)
        ])

        # cross-attention stack on the gated features
        self.layers = nn.ModuleList([
            CrossAttentionBlock(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )

    def forward(self, image_patches, image_patches_layer4,
                image_patches_layer8, text_feat):
        # normalize each layer's patches
        p4  = self.layer_norms[0](image_patches_layer4)    # (B, 196, 512)
        p8  = self.layer_norms[1](image_patches_layer8)    # (B, 196, 512)
        p12 = self.layer_norms[2](image_patches)           # (B, 196, 512)

        # text-conditioned gating
        gate_weights = torch.softmax(self.gate(text_feat), dim=-1)  # (B, 3)
        g = gate_weights.unsqueeze(-1).unsqueeze(-1)                # (B, 3, 1, 1)

        # stack and weight
        stacked = torch.stack([p4, p8, p12], dim=1)        # (B, 3, 196, 512)
        fused_patches = (stacked * g).sum(dim=1)            # (B, 196, 512)

        # cross-attention with text query
        query = text_feat.unsqueeze(1)  # (B, 1, 512)
        for layer in self.layers:
            query = layer(query, fused_patches)

        out = query.squeeze(1)          # (B, 512)
        return self.classifier(out)     # (B, 2)


def pool_patches(patches, target_h, target_w):
    """Pool patch grid to a coarser spatial resolution.

    Reshapes (B, N, D) to a 2D grid (assuming square), applies
    adaptive_avg_pool2d, and flattens back.
    Works with any square patch count (49 for ViT-B/32, 196 for ViT-B/16).
    """
    B, N, D = patches.shape
    grid_size = int(N ** 0.5)
    x = patches.reshape(B, grid_size, grid_size, D)
    x = x.permute(0, 3, 1, 2).float()                  # (B, D, g, g)
    # Use interpolate instead of adaptive_avg_pool2d for MPS compatibility
    # (MPS requires input divisible by output, which fails for e.g. 7→2)
    x = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=False)
    x = x.permute(0, 2, 3, 1).reshape(B, target_h * target_w, D)
    return x


class MultiGlimpse(nn.Module):
    """Multi-Glimpse Hierarchical Attention.

    Models the human visual search process:
      Glimpse 1 (coarse 2x2):  scan the whole scene  (4 regions)
      Glimpse 2 (medium 4x4):  focus on relevant regions (16 regions)
      Glimpse 3 (fine 7x7):    examine details (49 patches, full res)

    Each glimpse refines the query for the next scale. A single shared
    CrossAttentionBlock is reused across all scales (weight sharing →
    parameter-efficient, forces scale-invariant attention patterns).

    Default scales are for ViT-B/32 (7x7=49 patches). For ViT-B/16
    (14x14=196 patches), pass scales=[(4,4),(7,7),(14,14)].
    """
    REQUIRED_INPUTS = ("image_patches", "text_tokens", "text_mask")

    def __init__(self, embed_dim=512, num_heads=8, dropout=0.3,
                 num_classes=2, scales=None):
        super().__init__()

        if scales is None:
            scales = [(2, 2), (4, 4), (7, 7)]
        self.scales = scales

        # shared cross-attention block across all scales
        self.shared_attn = CrossAttentionBlock(embed_dim, num_heads, dropout)

        # per-scale LayerNorm for residual aggregation
        self.scale_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in scales
        ])

        self.output_norm = nn.LayerNorm(embed_dim)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )

    def forward(self, image_patches, text_tokens, text_mask):
        # image_patches: (B, 49, 512) for ViT-B/32
        # text_tokens:   (B, 77, 512)
        # text_mask:     (B, 77)

        query = text_tokens                              # (B, 77, 512)
        mask_expanded = text_mask.unsqueeze(-1).float()  # (B, 77, 1)
        grid_size = int(image_patches.shape[1] ** 0.5)  # 7 for ViT-B/32

        glimpse_outputs = []

        for i, (h, w) in enumerate(self.scales):
            # pool patches to this scale (skip if already at native res)
            if h == grid_size and w == grid_size:
                scale_patches = image_patches
            else:
                scale_patches = pool_patches(image_patches, h, w)

            # cross-attention: query attends to this scale's patches
            query = self.shared_attn(query, scale_patches)  # (B, 77, 512)

            # masked mean pool this glimpse's output
            pooled = (query * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            glimpse_outputs.append(self.scale_norms[i](pooled))  # (B, 512)

        # aggregate all glimpses via residual sum
        aggregated = sum(glimpse_outputs)       # (B, 512)
        aggregated = self.output_norm(aggregated)

        return self.classifier(aggregated)      # (B, 2)