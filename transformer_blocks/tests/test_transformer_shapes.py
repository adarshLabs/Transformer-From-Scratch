from pathlib import Path
import sys

import torch

try:
    from attention.masking import causal_mask
    from transformer_blocks.transformer_encoder_block import (
        TransformerEncoderBlock,
    )
    from transformer_blocks.transformer_decoder_block import (
        TransformerDecoderBlock,
    )
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from attention.masking import causal_mask
    from transformer_blocks.transformer_encoder_block import (
        TransformerEncoderBlock,
    )
    from transformer_blocks.transformer_decoder_block import (
        TransformerDecoderBlock,
    )


def test_encoder_block_shapes():

    # =================================================
    # Config
    # =================================================

    batch_size = 2
    seq_len = 10
    d_model = 64
    num_heads = 8

    # =================================================
    # Dummy Input
    # =================================================

    x = torch.randn(
        batch_size,
        seq_len,
        d_model,
    )

    # =================================================
    # Encoder Block
    # =================================================

    encoder_block = TransformerEncoderBlock(
        embed_dim=d_model,
        num_heads=num_heads,
    )

    # =================================================
    # Forward Pass
    # =================================================

    output = encoder_block(x, attention_mask=None)

    # =================================================
    # Expected Shape
    # =================================================

    expected_shape = (
        batch_size,
        seq_len,
        d_model,
    )

    # =================================================
    # Assertions
    # =================================================

    assert output.shape == expected_shape

    print("\nEncoder Block")
    print(f"Input Shape  : {x.shape}")
    print(f"Output Shape : {output.shape}")

    print("Encoder shape test passed.")


def test_decoder_block_shapes():

    # =================================================
    # Config
    # =================================================

    batch_size = 2
    seq_len = 10
    d_model = 64
    num_heads = 8

    # =================================================
    # Dummy Inputs
    # =================================================

    decoder_input = torch.randn(
        batch_size,
        seq_len,
        d_model,
    )

    encoder_output = torch.randn(
        batch_size,
        seq_len,
        d_model,
    )

    # =================================================
    # Causal Mask
    # =================================================

    causal_masking = causal_mask(seq_len)

    # =================================================
    # Decoder Block
    # =================================================

    decoder_block = TransformerDecoderBlock(
        embed_dim=d_model,
        num_heads=num_heads,
    )

    # =================================================
    # Forward Pass
    # =================================================

    output = decoder_block(
        x=decoder_input,
        encoder_output=encoder_output,
        self_attention_mask=causal_masking,
        cross_attention_mask=None
    )

    # =================================================
    # Expected Shape
    # =================================================

    expected_shape = (
        batch_size,
        seq_len,
        d_model,
    )

    # =================================================
    # Assertions
    # =================================================

    assert output.shape == expected_shape

    print("\nDecoder Block")
    print(f"Input Shape  : {decoder_input.shape}")
    print(f"Output Shape : {output.shape}")

    print("Decoder shape test passed.")


def test_no_nan_values():

    # =================================================
    # Config
    # =================================================

    batch_size = 2
    seq_len = 10
    d_model = 64
    num_heads = 8

    # =================================================
    # Dummy Inputs
    # =================================================

    x = torch.randn(
        batch_size,
        seq_len,
        d_model,
    )

    encoder_output = torch.randn(
        batch_size,
        seq_len,
        d_model,
    )

    # =================================================
    # Blocks
    # =================================================

    encoder_block = TransformerEncoderBlock(
        embed_dim=d_model,
        num_heads=num_heads,
    )

    decoder_block = TransformerDecoderBlock(
        embed_dim=d_model,
        num_heads=num_heads,
    )

    # =================================================
    # Encoder Forward
    # =================================================

    encoder_out = encoder_block(x, attention_mask=None)

    # =================================================
    # Decoder Forward
    # =================================================

    causal_masking = causal_mask(seq_len)

    decoder_out = decoder_block(
        x=x,
        encoder_output=encoder_output,
        self_attention_mask=causal_masking,
        cross_attention_mask=None
    )

    # =================================================
    # NaN Assertions
    # =================================================

    assert not torch.isnan(encoder_out).any()

    assert not torch.isnan(decoder_out).any()

    print("\nNaN Tests")
    print("Encoder NaN check passed.")
    print("Decoder NaN check passed.")


if __name__ == "__main__":

    print("=" * 60)
    print("Running Transformer Block Tests")
    print("=" * 60)

    test_encoder_block_shapes()

    test_decoder_block_shapes()

    test_no_nan_values()

    print("\nAll transformer tests passed successfully.")
