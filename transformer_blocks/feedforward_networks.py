import torch
import torch.nn as nn

class FeedForwardNetwork(nn.Module):
    def __init(self, embed_dim, expansion_factor = 4, dropout=0.1):
        super().__init__()
        hidden_dim = expansion_factor* embed_dim
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x):
        return self.ffn(x)