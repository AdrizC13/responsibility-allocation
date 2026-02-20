import torch # type: ignore
import torch.nn as nn # type: ignore

class TemporalEncoder(nn.Module):
    """Encodes time‑series of a single car using a robust Bidirectional LSTM."""

    def __init__(self, input_dim, embed_dim):
        super().__init__()
        # We use a hidden_dim that is half the desired embed_dim 
        # because bidirectional will double it (hidden_dim * 2).
        hidden_dim = embed_dim // 2
        
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=1, 
            batch_first=True, 
            bidirectional=True
        )
        
        # Ensures the final output is exactly embed_dim
        self.projection = nn.Linear(hidden_dim * 2, embed_dim)
        
        # Initialize weights for better convergence
        self._init_weights()

    def _init_weights(self):
        # Orthogonal initialization helps prevent exploding/vanishing gradients in RNNs
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)

    def forward(self, X):
        """
        X: Tensor of shape (1000, 9)
        """
        # Add batch dimension: (1, 1000, 9)
        X = X.unsqueeze(0)  
        
        # h shape: (num_layers * num_directions, batch, hidden_dim)
        _, (h, _) = self.lstm(X)
        
        # Concatenate the final forward hidden state and the final backward hidden state
        # h[-2] is the last hidden state of the forward pass
        # h[-1] is the last hidden state of the backward pass
        combined_h = torch.cat((h[-2,:,:], h[-1,:,:]), dim=1)
        
        # Project back to the desired embed_dim and remove batch dimension
        return self.projection(combined_h).squeeze(0) # Output shape: (embed_dim)
