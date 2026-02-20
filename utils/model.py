import torch.nn as nn
from utils.temporal_encoder import TemporalEncoder
from utils.attention import VehicleAttention
from utils.classifier import ScenarioClassifier

class AccidentModel(nn.Module):
    """Full end‑to‑end model."""

    def __init__(self, input_dim, embed_dim, attn_dim, num_cars):
        super().__init__()
        self.encoder = TemporalEncoder(input_dim, embed_dim)
        self.attention = VehicleAttention(embed_dim, attn_dim)
        self.classifier = ScenarioClassifier(num_cars, attn_dim)

    def forward(self, cars):
        embeddings = [self.encoder(X) for X in cars]
        V_prime = self.attention(embeddings)
        return self.classifier(V_prime)
