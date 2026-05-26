import torch
import torch.nn as nn

from attention.multi_head_attention import MultiHeadAttention
from transformer_blocks.feed_forward_network import FeedForwardNetwork


class TransformerDecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, expansion_factor=4, dropout=0.1):
        super().__init__()

        self.self_attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardNetwork(embed_dim, expansion_factor, dropout)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, self_attention_mask, cross_attention_mask):

        # Self Attention
        residual = x
        x = self.norm1(x)
        attention_output, _ = self.self_attention(query=x, mask=self_attention_mask)
        x = residual + self.dropout(attention_output)

        # Cross Attention
        residual = x
        x = self.norm2(x)
        cross_attention_output, _ = self.cross_attention(
            query=x,
            key=encoder_output,
            value=encoder_output,
            mask=cross_attention_mask,
        )
        x = residual + self.dropout(cross_attention_output)

        # Feed Forward Network
        residual = x
        x = self.norm3(x)
        ffn_output = self.ffn(x)
        x = residual + self.dropout(ffn_output)

        return x
