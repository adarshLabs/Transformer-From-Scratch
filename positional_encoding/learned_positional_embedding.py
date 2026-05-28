import torch
import torch.nn as nn

class LearnedPositionalEmbedding(nn.Module):

    def __init__(self, embed_dim, max_seq_len=512):
        super().__init__()

        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)

    def forward(self, x):
        _, seq_len, _ = x.shape

        position = torch.arange(0, seq_len, device=x.device).unsqueeze(0)

        positional_embedding = self.position_embedding(position)

        return x + positional_embedding
    
def main():

    batch_size = 2
    seq_len = 16
    max_seq_len = 5000
    embed_dim = 128
    x = torch.rand(batch_size, seq_len, embed_dim)

    positional_embedding = LearnedPositionalEmbedding(max_seq_len, embed_dim)
    output = positional_embedding(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"positional_embedding shape: {positional_embedding.position_embedding.weight.shape}")

if __name__=="__main__":
    main()
