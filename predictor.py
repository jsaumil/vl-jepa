import torch
import torch.nn as nn

from utils.modules import CrossAttentionBlock

class Predictor(nn.Module):
    def __init__(self,embed_dim = 768,depth=6,num_heads=12,mlp_ratio=4.0):
        super().__init__()

        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads=num_heads
        self.mlp_ratio = mlp_ratio

        self.predictor_embed = nn.Linear(embed_dim,embed_dim,bias=True)

        self.predictor = nn.ModuleList([
            CrossAttentionBlock(
                dim = self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
            )
            for i in range(self.depth)
        ])

        self.norm = nn.LayerNorm(self.embed_dim)
        self.pred_proj = nn.Linear(embed_dim,embed_dim,bias=True)

    def forward(self,embed_vit, q):
        x = embed_vit
        y = q
        for blk in self.predictor:
            x = blk(x,y)
        x = self.norm(x)

        x = self.pred_proj(x)

        return x