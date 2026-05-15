import torch

def causal_mask(seq_len, device=None):

    mask = torch.tril(torch.ones((seq_len, seq_len), device=device))

    #shape should be (1, 1, seq_len, seq_len)
    return mask.bool().unsqueeze(0).unsqueeze(1)

def padding_mask(input_ids, padding_token=0):

    mask = (input_ids != padding_token)

    #shape should be (batch_size, 1, 1, dim)
    return mask.unsqueeze(1).unsqueeze(2)


def combined_mask(causal_mask, padding_mask):
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
