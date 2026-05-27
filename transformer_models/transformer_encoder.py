import torch
import torch.nn as nn
from positional_encoding.rotary_positional_embedding import RotaryPositionalEmbedding
from transformer_blocks.transformer_encoder_block import TransformerEncoderBlock


class TranformerEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, expansion_factor=4, dropout=0.1, positional_encoding_type="rope"):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        self.qk_positional_encoding = None
        if positional_encoding_type=="rope":
            head_dim = embed_dim // num_heads
            self.qk_positional_encoding = RotaryPositionalEmbedding(
                head_dim
            ).apply_rotary

        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    embed_dim=embed_dim,
                    num_head=num_heads,
                    expansion_factor=expansion_factor,
                    dropout=dropout,
                    qk_positional_encoding=self.qk_positional_encoding
                )
                for _ in range(num_layers)
            ]
        )

        self.attention = TransformerEncoderBlock( embed_dim, num_heads, qk)
