import torch # type: ignore
import torch.nn as nn # type: ignore

class ScenarioClassifier(nn.Module):
    def __init__(self, num_cars, attn_dim):
        super().__init__()
        
        # Multi-layer attention for better feature selection
        self.attention = nn.Sequential(
            nn.Linear(attn_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # Deep network with LayerNorm (Safe for batch size 1)
        self.net = nn.Sequential(
            nn.Linear(attn_dim, 1024),
            nn.LayerNorm(1024), 
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            
            nn.Linear(256, num_cars)
        )

    def forward(self, V_prime):
        # 1. Attention Pooling
        scores = self.attention(V_prime)     # (N, 1)
        alpha = torch.softmax(scores, dim=0) # (N, 1)
        scenario_embedding = torch.sum(alpha * V_prime, dim=0) # (attn_dim,)
        
        # 2. Add Batch Dim for the network layers: (attn_dim,) -> (1, attn_dim)
        x = scenario_embedding.unsqueeze(0)
        
        # 3. Forward through MLP
        logits = self.net(x)
        
        # 4. Remove Batch Dim for the loss function: (1, num_cars) -> (num_cars,)
        return logits.squeeze(0)
