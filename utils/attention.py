import torch
import torch.nn as nn
import math

class VehicleAttention(nn.Module):
    """Dot‑product attention across vehicles."""

    def __init__(self, embed_dim, attn_dim):
        super().__init__()
        self.WQ = nn.Linear(embed_dim, attn_dim)
        self.WK = nn.Linear(embed_dim, attn_dim)
        self.WV = nn.Linear(embed_dim, attn_dim)

    def forward(self, embeddings):
        # embeddings: list of (D)
        Q = torch.stack([self.WQ(e) for e in embeddings])  # (C, H)
        K = torch.stack([self.WK(e) for e in embeddings])  # (C, H)
        V = torch.stack([self.WV(e) for e in embeddings])  # (C, H)
        d_k = Q.size(-1)
        scores = Q @ K.T / math.sqrt(d_k)
        weights = torch.softmax(scores, dim=1)
        return weights @ V