import torch
import torch.nn as nn
from positional_encoding.rotary_positional_embedding import RotaryPositionalEmbedding
from positional_encoding.learned_positional_embedding import LearnedPositionalEmbedding
from positional_encoding.sinusoidal_positional_encoding import SinusoidalPositionalEncoding
from transformer_blocks.transformer_encoder_block import TransformerEncoderBlock


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        num_heads,
        num_layers,
        max_seq_len=512,
        expansion_factor=4,
        dropout=0.1,
        positional_encoding_type="rope",
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        self.qk_positional_encoding = None
        if positional_encoding_type == "rope":
            head_dim = embed_dim // num_heads
            self.qk_positional_encoding = RotaryPositionalEmbedding(
                head_dim
            )
            self.input_positional_encoding = None

        elif positional_encoding_type == "learned":
            self.qk_positional_encoding = None
            self.input_positional_encoding = LearnedPositionalEmbedding(
                embed_dim=embed_dim,
                max_seq_len=max_seq_len,
            )

        elif positional_encoding_type == "sinusoidal":
            self.qk_positional_encoding = None
            self.input_positional_encoding = SinusoidalPositionalEncoding(
                embed_dim=embed_dim,
                max_seq_len=max_seq_len,
            )

        else:
            self.qk_positional_encoding = None
            self.input_positional_encoding = None

        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    expansion_factor=expansion_factor,
                    dropout=dropout,
                    qk_positional_encoding=self.qk_positional_encoding
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, input_ids, attention_mask=None):
        x = self.token_embedding(input_ids)

        if self.input_positional_encoding is not None:
            x = self.input_positional_encoding(x)

        for layer in self.layers:
            x = layer(x, attention_mask)

        return x
