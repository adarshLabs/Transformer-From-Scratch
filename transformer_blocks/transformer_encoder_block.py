import torch
import torch.nn as nn
from attention.multi_head_attention import MultiHeadAttention
from transformer_blocks.feed_forward_network import FeedForwardNetwork
from positional_encoding.rotary_positional_embedding import RotaryPositionalEmbedding

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, expansion_factor=4, dropout=0.1):
        super().__init__()

        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardNetwork(embed_dim, expansion_factor, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, attention_mask):
        residual = x
        x = self.norm1(x)
        attention_output, _ = self.attention(query=x, mask=attention_mask)
        x = residual + self.dropout(attention_output)
        x = self.norm2(x)
        residual = x
        ffn_output = self.ffn(residual)
        x = residual  + self.dropout(ffn_output)
        return x
    

def main():
    batch_size=2
    seq_len=10
    d_model=64
    num_heads = 4


if __name__=="__main__":
    main()