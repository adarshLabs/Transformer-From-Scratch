import torch
import torch.nn as nn

try:
    from .scaled_dot_product_attention import ScaledDotProductAttention
except ImportError:
    from scaled_dot_product_attention import ScaledDotProductAttention

class MultiHeadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0.1, qk_positional_encoding=None):
        super().__init__()
        assert embed_dim % num_heads==0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = (self.embed_dim // self.num_heads)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.qk_positional_encoding = qk_positional_encoding

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attention = ScaledDotProductAttention(dropout)

    def split_head(self, x):
        B, S, E = x.shape
        x = x.view(B, S, self.num_heads, self.head_dim)
        x = x.transpose(1, 2)
        return x
    
    def combine_heads(self, x):
        B, H, S, D = x.shape

        x = x.transpose(1, 2)
        x = x.contiguous().view(B, S, self.embed_dim)
        return x
    
    def forward(self, query, key=None, value=None, mask = None):

        if key is None:
            key = query
        if value is None:
            value = key

        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        Q_split = self.split_head(Q)
        K_split = self.split_head(K)
        V_split = self.split_head(V)

        if self.qk_positional_encoding is not None:
            Q_split, K_split = self.qk_positional_encoding.apply_rotary(Q_split, K_split)

        attn_output, attn_weights =  self.attention(Q_split, K_split, V_split, mask)

        concatenated_output = self.combine_heads(attn_output)
        output = self.out_proj(concatenated_output)

        return output, attn_weights


def main():
    x = torch.randn(2, 16, 128)

    mha = MultiHeadAttention(

        embed_dim=128,

        num_heads=8

    )

    output, attn = mha(x)

    assert output.shape == (2, 16, 128)
    assert attn.shape == (2, 8, 16, 16)
    
if __name__=="__main__":
    main()
