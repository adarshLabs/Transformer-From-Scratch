from pathlib import Path
import sys

import torch
import torch.nn as nn

try:
    from attention.masking import causal_mask, combined_mask, padding_mask
    from positional_encoding.learned_positional_embedding import LearnedPositionalEmbedding
    from transformer_blocks.transformer_encoder_block import TransformerEncoderBlock
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from attention.masking import causal_mask, combined_mask, padding_mask
    from positional_encoding.learned_positional_embedding import LearnedPositionalEmbedding
    from transformer_blocks.transformer_encoder_block import TransformerEncoderBlock


def test_transformer_encoder_pipeline_forward_and_backward():
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 6
    vocab_size = 32
    embed_dim = 16
    num_heads = 4
    padding_token = 0

    input_ids = torch.tensor(
        [
            [5, 7, 9, 11, 0, 0],
            [3, 4, 8, 12, 16, 20],
        ]
    )

    token_embedding = nn.Embedding(vocab_size, embed_dim)
    positional_embedding = LearnedPositionalEmbedding(seq_len, embed_dim)
    encoder_block = TransformerEncoderBlock(
        embed_dim=embed_dim,
        num_heads=num_heads,
        expansion_factor=2,
        dropout=0.0,
    )

    token_embeddings = token_embedding(input_ids)
    x = positional_embedding(token_embeddings)

    mask = combined_mask(
        causal_mask(seq_len, device=input_ids.device),
        padding_mask(input_ids, padding_token=padding_token),
    )

    output = encoder_block(x, mask)

    assert output.shape == (batch_size, seq_len, embed_dim)
    assert torch.isfinite(output).all()

    loss = output.sum()
    loss.backward()

    assert token_embedding.weight.grad is not None
    assert torch.isfinite(token_embedding.weight.grad).all()

    for name, parameter in encoder_block.named_parameters():
        assert parameter.grad is not None, f"Missing gradient for {name}"
        assert torch.isfinite(parameter.grad).all(), f"Invalid gradient for {name}"


def main():
    test_transformer_encoder_pipeline_forward_and_backward()
    print("Transformer encoder pipeline forward/backward test passed!")


if __name__ == "__main__":
    main()
