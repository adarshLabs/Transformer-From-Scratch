import torch
import torch.nn as nn


def rotate_half(x):
    # Input x shape: (..., D), where D is even.
    even = x[..., ::2]
    odd = x[..., 1::2]

    # Stacked shape: (..., D / 2, 2)
    output = torch.stack((-odd, even), dim=-1)

    # Output shape: (..., D)
    return output.flatten(-2)

class RotaryPositionalEmbedding(nn.Module):

    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        # inv_freq shape: (D / 2)
        inv_freq= 1.0/ (base ** (torch.arange(0, dim, 2).float()/dim))
        self.register_buffer( "inv_freq", inv_freq)

    def get_cos_sin(self, seq_len, device):
        # positions shape: (S)
        positions = torch.arange(0, seq_len, device=device).float()

        # freq shape: (S, D / 2), emb shape after repeat: (S, D)
        freq = torch.outer(positions, self.inv_freq)
        emb = torch.repeat_interleave(freq, 2, dim=-1)
        # cos and sin shapes: (1, 1, S, D), broadcast over batch and heads.
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        return cos, sin
    
    def apply_rotary(self, q, k):
        # Input q and k shapes: (B, H, S, D)

        seq_len = q.shape[2]

        cos, sin = self.get_cos_sin(seq_len, q.device)

        q_rot = q * cos + (rotate_half(q) * sin)
        k_rot = k* cos + (rotate_half(k) * sin)

        # Output q_rot and k_rot shapes: (B, H, S, D)
        return q_rot, k_rot
