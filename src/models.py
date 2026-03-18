from turtle import forward
import torch
import torch.nn as nn

class ConcatMLP(nn.Module):
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