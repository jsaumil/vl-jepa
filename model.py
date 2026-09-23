import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from x_encoder import VisionTransformer
from y_encoder import Transformer
from predictor import Predictor
from utils.patch_embed import PatchEmbed3D

class DeepFake(nn.Module):
    def __init__(self, max_k=2000,embed_dim=768,vocab_size=50244, depth=12, num_heads=12, pred_depth=6, pred_heads=12, **kwargs):
        super().__init__()

        self.x_encoder = VisionTransformer(embed_dim=embed_dim, depth=depth, num_heads=num_heads)
        self.y_encoder = Transformer(embed_dim=embed_dim, depth=depth, num_heads=num_heads)
        self.predictor = Predictor(embed_dim=embed_dim, depth=pred_depth, num_heads=pred_heads)
        self.embed = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim
        )
        
        self.max_k = max_k
        self.embed_dim = embed_dim
        self.patcher = PatchEmbed3D(embed_dim=embed_dim)
        # fix the positional embeddings
        self.pos = nn.Parameter(
            torch.zeros(1,self.max_k, self.embed_dim)
        )
        self.text_pos = nn.Parameter(
            torch.zeros(1, self.max_k, embed_dim)
        )
        # CLIP-style learnable temperature, clamped so logits can't blow up
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def forward(self, x, query, y, train=False):
        x = self.patcher(x)
        B, N, C = x.shape
        x = x + self.pos[:, :N, :]
        x = self.x_encoder(x)

        q = self.embed(query)
        Bq, Q, Cq = q.shape
        q = q + self.text_pos[:, :Q, :]
        y_pred = self.predictor(x,q)
        loss = None
        if train:
            y = self.embed(y)
            By, Y, Cy = y.shape
            y = y + self.pos[:, :Y, :]
            y = self.y_encoder(y)

            y_pred = y_pred.mean(dim=1) # [B, C]
            y = y.mean(dim=1) # [B, C]

            y_pred_n = F.normalize(y_pred, dim=-1)
            y_n = F.normalize(y, dim=-1)
            logit_scale = self.logit_scale.clamp(max=math.log(100)).exp()
            logits = logit_scale * (y_pred_n @ y_n.transpose(-1, -2))
            labels = torch.arange(logits.shape[0], device=logits.device)
            loss_i2t = F.cross_entropy(logits, labels)
            loss_t2i = F.cross_entropy(logits.t(), labels)
            loss = (loss_i2t + loss_t2i) / 2

            return y_pred, loss

        return y_pred