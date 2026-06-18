import torch

def causal_mask(seq_len, device=None):

    # Base mask shape: (S, S), where future positions are zero.
    mask = torch.tril(torch.ones((seq_len, seq_len), device=device))

    # Output shape: (1, 1, S, S), broadcast over batch and heads.
    return mask.bool().unsqueeze(0).unsqueeze(1)

def padding_mask(input_ids, padding_token=0):
    # input_ids shape: (B, S)

    # Base mask shape: (B, S), with True for non-padding tokens.
    mask = (input_ids != padding_token)

    # Output shape: (B, 1, 1, S), broadcast over heads and query positions.
    return mask.unsqueeze(1).unsqueeze(2)


def combined_mask(causal_mask, padding_mask):
    # causal_mask shape: (1, 1, S, S)
    # padding_mask shape: (B, 1, 1, S)
    # Output shape: (B, 1, S, S)
    return causal_mask & padding_mask


def main():
    seq_len= 5

    causal = causal_mask(seq_len)

    input_ids = torch.tensor([
        [5, 7, 2, 0, 0],
        [1, 9, 4, 8, 3]
    ])

    padding = padding_mask(input_ids)
    combined = combined_mask(causal, padding)

    print("Causal mask shape: ", causal.shape)
    print("Padding mask shape: ", padding.shape)
    print("Combined mask shape: ", combined.shape)

    #output, attn = scaled_dot_product_attention(Q, K, V, mask)


if __name__=="__main__":
    main()
