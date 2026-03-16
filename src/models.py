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

        # WHY MultiheadAttention?
        # it lets text query the image across multiple "perspectives" simultaneously
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm = nn.LayerNorm(embed_dim)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )

    def forward(self, image_feat, text_feat):
        # unsqueeze to add sequence dimension — attention expects (B, seq_len, dim)
        img = image_feat.unsqueeze(1)   # (B, 1, 512)
        txt = text_feat.unsqueeze(1)    # (B, 1, 512)

        # text queries the image
        # query=txt, key=img, value=img
        # WHY? the question decides what to look for, image provides the answers
        attended, _ = self.cross_attn(query=txt, key=img, value=img)

        # residual connection + norm — standard transformer trick for stable training
        out = self.norm(attended.squeeze(1) + text_feat)

        return self.classifier(out)     # (B, 2)        