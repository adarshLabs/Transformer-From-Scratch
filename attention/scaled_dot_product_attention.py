import torch
import torch.nn.functional as F
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        # Q shape: (B, H, S_q, D)
        # K and V shapes: (B, H, S_k, D)
        # mask shape: broadcastable to (B, H, S_q, S_k).
        dim = Q.size(-1)
        # scores shape: (B, H, S_q, S_k)
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scaling_factor = dim ** 0.5
        scores = scores/scaling_factor
        
        if mask is not None:
            scores = scores.masked_fill(mask==0, float('-inf'))

        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        # output shape: (B, H, S_q, D)
        output = torch.matmul(attention_weights, V)
        # attention_weights shape: (B, H, S_q, S_k)
        return output, attention_weights
        

def main():
    batch_size = 10
    heads = 20
    seq_len = 25
    dim = 40

    # Demo tensor shapes: (batch_size, heads, seq_len, dim)
    Q = torch.randn((batch_size, heads, seq_len, dim))
    K = torch.randn((batch_size, heads, seq_len, dim))
    V = torch.randn((batch_size, heads, seq_len, dim))

    attention = ScaledDotProductAttention()

    output, attn_weights = attention.forward(Q, K, V)
    print("Output shape: ", output.shape)
    print("Attention_weights: ", attn_weights.shape)

    print(torch.sum(attn_weights, dim=-1))
    return 


if __name__=="__main__":
    main()
