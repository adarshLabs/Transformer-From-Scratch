# import torch
# import torch.nn as nn

# try:
#     from .scaled_dot_product_attention import ScaledDotProductAttention
# except ImportError:
#     from scaled_dot_product_attention import ScaledDotProductAttention


# class MultiHeadAttention(nn.Module):

#     def __init__(self, embed_dim, num_heads, dropout=0.1, qk_positional_encoding=None):
#         super().__init__()
#         assert embed_dim % num_heads == 0
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.dropout = dropout
#         # Per-head dimension D = E / H.
#         self.head_dim = self.embed_dim // self.num_heads
#         self.q_proj = nn.Linear(embed_dim, embed_dim)
#         self.k_proj = nn.Linear(embed_dim, embed_dim)
#         self.v_proj = nn.Linear(embed_dim, embed_dim)

#         self.qk_positional_encoding = qk_positional_encoding

#         self.out_proj = nn.Linear(embed_dim, embed_dim)
#         self.attention = ScaledDotProductAttention(dropout)

#     def split_head(self, x):
#         # Input x shape: (B, S, E)
#         B, S, E = x.shape
#         # Reshaped x shape: (B, S, H, D)
#         x = x.view(B, S, self.num_heads, self.head_dim)
#         # Output x shape: (B, H, S, D)
#         x = x.transpose(1, 2)
#         return x

#     def combine_heads(self, x):
#         # Input x shape: (B, H, S, D)
#         B, H, S, D = x.shape

#         # Output x shape: (B, S, E)
#         x = x.transpose(1, 2)
#         x = x.contiguous().view(B, S, self.embed_dim)
#         return x

#     def forward(self, query, key=None, value=None, mask=None, past_kv=None):
#         # query shape: (B, S_q, E)
#         # key/value shapes: (B, S_k, E); default to query for self-attention.
#         # mask shape: broadcastable to (B, H, S_q, S_k).

#         if key is None:
#             key = query
#         if value is None:
#             value = key

#         Q = self.q_proj(query)
#         K = self.k_proj(key)
#         V = self.v_proj(value)

#         # Split projection shapes: (B, H, S_q, D), (B, H, S_k, D), (B, H, S_k, D)
#         Q_split = self.split_head(Q)
#         K_split = self.split_head(K)
#         V_split = self.split_head(V)

#         past_len = 0
#         if past_kv is not None:
#             K_cache, V_cache = past_kv
#             past_len = K_cache.shape[2]

#         if self.qk_positional_encoding is not None:
#             Q_split, K_split = self.qk_positional_encoding.apply_rotary(
#                 Q_split, K_split, offset=past_len
#             )

#         if past_kv is not None:
#             K_split = torch.cat([K_cache, K_split], dim=2)
#             V_split = torch.cat([V_cache, V_split], dim=2)

#         new_kv = (K_split, V_split)

#         # attn_output shape: (B, H, S_q, D), attn_weights shape: (B, H, S_q, S_k)
#         attn_output, attn_weights = self.attention(Q_split, K_split, V_split, mask)

#         concatenated_output = self.combine_heads(attn_output)
#         output = self.out_proj(concatenated_output)

#         # output shape: (B, S_q, E)
#         return output, attn_weights, new_kv


# def main():
#     # Demo input shape: (B, S, E)
#     x = torch.randn(2, 16, 128)

#     mha = MultiHeadAttention(embed_dim=128, num_heads=8)

#     output, attn = mha(x)

#     assert output.shape == (2, 16, 128)
#     assert attn.shape == (2, 8, 16, 16)


# if __name__ == "__main__":
#     main()
import torch
import torch.nn as nn

try:
    from .scaled_dot_product_attention import ScaledDotProductAttention
except ImportError:
    from scaled_dot_product_attention import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0.1, qk_positional_encoding=None):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        # Per-head dimension D = E / H.
        self.head_dim = self.embed_dim // self.num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.qk_positional_encoding = qk_positional_encoding

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attention = ScaledDotProductAttention(dropout)

    def split_head(self, x):
        # Input x shape: (B, S, E)
        B, S, E = x.shape
        # Reshaped x shape: (B, S, H, D)
        x = x.view(B, S, self.num_heads, self.head_dim)
        # Output x shape: (B, H, S, D)
        x = x.transpose(1, 2)
        return x

    def combine_heads(self, x):
        # Input x shape: (B, H, S, D)
        B, H, S, D = x.shape
        # Output x shape: (B, S, E)
        x = x.transpose(1, 2)
        x = x.contiguous().view(B, S, self.embed_dim)
        return x

    def forward(self, query, key=None, value=None, mask=None, past_kv=None):
        """
        query  shape: (B, S_q, E)
        key    shape: (B, S_k, E) — defaults to query for self-attention
        value  shape: (B, S_k, E) — defaults to key
        mask   shape: broadcastable to (B, H, S_q, S_k)
        past_kv: tuple (K_cache, V_cache) with shapes (B, H, S_past, D)
                 None during prefill, set during decode

        Returns:
            output      : (B, S_q, E)
            attn_weights: (B, H, S_q, S_k)
            new_kv      : (K, V) tuple — full cache including current step
        """
        if key is None:
            key = query
        if value is None:
            value = key

        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        # Split into heads: (B, S, E) → (B, H, S, D)
        Q_split = self.split_head(Q)
        K_split = self.split_head(K)
        V_split = self.split_head(V)

        # ── KV CACHE: get past length for RoPE offset ──────────────────
        # past_len tells RoPE where in the sequence the current tokens sit.
        # During prefill: past_len=0, positions start from 0.
        # During decode:  past_len=N, new token gets position N.
        past_len = 0
        if past_kv is not None:
            K_cache, V_cache = past_kv
            past_len = K_cache.shape[2]   # sequence dimension of cached K

        # ── APPLY RoPE BEFORE cache concat ─────────────────────────────
        # RoPE is applied to Q and K only (not V).
        # offset=past_len ensures the new token(s) get correct absolute positions.
        # Cached K already has RoPE applied from when those tokens were processed.
        if self.qk_positional_encoding is not None:
            Q_split, K_split = self.qk_positional_encoding.apply_rotary(
                Q_split, K_split, offset=past_len
            )

        # ── CONCAT CACHE AFTER RoPE ─────────────────────────────────────
        # Concatenate cached K,V with current K,V along sequence dimension.
        # During prefill: past_kv is None, no concat happens.
        # During decode:  K grows from (B,H,1,D) to (B,H,past_len+1,D)
        if past_kv is not None:
            K_split = torch.cat([K_cache, K_split], dim=2)
            V_split = torch.cat([V_cache, V_split], dim=2)

        # Store full K, V (cache + current) for caller to save and pass back
        new_kv = (K_split, V_split)

        # ── ATTENTION ────────────────────────────────────────────────────
        # During decode: Q shape (B,H,1,D), K shape (B,H,past_len+1,D)
        # scores shape:  (B,H,1,past_len+1) — single query attends to full cache
        # mask is None during decode — single query is always the last token
        # During prefill: standard (B,H,T,T) attention with causal mask
        attn_output, attn_weights = self.attention(Q_split, K_split, V_split, mask)

        concatenated_output = self.combine_heads(attn_output)
        output = self.out_proj(concatenated_output)

        # output shape: (B, S_q, E)
        return output, attn_weights, new_kv


def main():
    # Demo input shape: (B, S, E)
    x = torch.randn(2, 16, 128)

    mha = MultiHeadAttention(embed_dim=128, num_heads=8)

    # forward() returns 3 values: output, attn_weights, new_kv
    output, attn_weights, new_kv = mha(x)   # ← fixed: was unpacking 2 values

    assert output.shape == (2, 16, 128), f"Expected (2,16,128), got {output.shape}"
    assert attn_weights.shape == (2, 8, 16, 16), f"Expected (2,8,16,16), got {attn_weights.shape}"
    assert isinstance(new_kv, tuple) and len(new_kv) == 2, "new_kv should be (K, V) tuple"

    K, V = new_kv
    assert K.shape == (2, 8, 16, 16), f"Cache K shape wrong: {K.shape}"
    assert V.shape == (2, 8, 16, 16), f"Cache V shape wrong: {V.shape}"

    print("MultiHeadAttention: all shape checks passed")
    print(f"  output     : {output.shape}")
    print(f"  attn_weights: {attn_weights.shape}")
    print(f"  K_cache    : {K.shape}")
    print(f"  V_cache    : {V.shape}")


if __name__ == "__main__":
    main()