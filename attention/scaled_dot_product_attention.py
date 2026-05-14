import torch
import torch.nn.functional as F

class ScaledDotProductAttention:
    def __init__(self):
        pass

    def forward(self, Q, K, V, mask=None):
        dim = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scaling_factor = dim**0.5
        scores = scores/scaling_factor
        
        if mask is not None:
            scores = scores.masked_fill(mask==0, float('-inf'))

        
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
        

def main():
    batch_size = 10
    heads = 20
    seq_len = 25
    dim = 40

    Q = torch.randn((batch_size, heads, seq_len, dim))
    K = torch.randn((batch_size, heads, seq_len, dim))
    V = torch.randn((batch_size, heads, seq_len, dim))

    attention = ScaledDotProductAttention()

    output, attn_weights = attention.forward(Q, K, V)
    print("Output shape: ", output.shape)
    print("Attention_weights: ", attn_weights.shape)

    return 


if __name__=="__main__":
    main()