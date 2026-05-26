from pathlib import Path
import sys

import torch
import torch.nn as nn

try:
    from attention.masking import padding_mask
    from positional_encoding.rotary_positional_embedding import RotaryPositionalEmbedding
    from transformer_blocks.transformer_encoder_block import TransformerEncoderBlock
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from attention.masking import padding_mask
    from positional_encoding.rotary_positional_embedding import RotaryPositionalEmbedding
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
    encoder_block = TransformerEncoderBlock(
        embed_dim=embed_dim,
        num_heads=num_heads,
        expansion_factor=2,
        dropout=0.0,
    )

    x = token_embedding(input_ids)

    mask = padding_mask(input_ids, padding_token=padding_token)

    normalized_x = encoder_block.norm1(x)
    q = encoder_block.attention.q_proj(normalized_x)
    k = encoder_block.attention.k_proj(normalized_x)
    v = encoder_block.attention.v_proj(normalized_x)

    q = encoder_block.attention.split_head(q)
    k = encoder_block.attention.split_head(k)
    v = encoder_block.attention.split_head(v)

    rotary_embedding = RotaryPositionalEmbedding(encoder_block.attention.head_dim)
    q_rot, k_rot = rotary_embedding.apply_rotary(q, k)

    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape
    assert torch.allclose(q.norm(dim=-1), q_rot.norm(dim=-1), atol=1e-5)
    assert torch.allclose(k.norm(dim=-1), k_rot.norm(dim=-1), atol=1e-5)

    attention_output, attention_weights = encoder_block.attention.attention(
        q_rot,
        k_rot,
        v,
        mask,
    )
    attention_output = encoder_block.attention.combine_heads(attention_output)
    attention_output = encoder_block.attention.out_proj(attention_output)

    assert attention_output.shape == (batch_size, seq_len, embed_dim)
    assert attention_weights.shape == (batch_size, num_heads, seq_len, seq_len)
    assert torch.all(attention_weights[0, :, :, 4:] == 0)

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
