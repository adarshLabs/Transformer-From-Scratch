# Transformer From Scratch

A from-scratch PyTorch implementation of transformer building blocks,
encoder/decoder modules, and a growing path toward a GPT-2 style decoder-only
language model. The repository currently focuses on attention, positional
encoding, and transformer block components, with small runnable demos and tests
for attention, encoder pipeline behavior, and decoder cross-attention behavior.

## Current Components

- Scaled dot-product attention
- Multi-head self-attention and cross-attention
- Causal, padding, and combined attention masks
- Sinusoidal positional encoding
- Learned positional embedding
- Rotary positional embedding
- Feed-forward network
- Transformer encoder block
- Transformer decoder block with self-attention and cross-attention
- Transformer encoder model with selectable positional encoding
- Test that compares the custom multi-head attention output with
  `torch.nn.MultiheadAttention`
- Transformer encoder pipeline test with learned positional embeddings, padding
  masks, forward pass, and backward pass checks
- Transformer decoder pipeline test with causal masking, cross-attention padding
  masks, forward pass, and backward pass checks

## Project Structure

```text
attention/
  masking.py
  multi_head_attention.py
  scaled_dot_product_attention.py
  tests/
    test_against_pytorch_mha.py
positional_encoding/
  learned_positional_embedding.py
  rotary_positional_embedding.py
  sinusoidal_positional_encoding.py
transformer_blocks/
  feed_forward_network.py
  transformer_decoder_block.py
  transformer_encoder_block.py
  tests/
    test_transformer_decoder_pipeline.py
    test_transformer_encoder_pipeline.py
transformer_models/
  transformer_encoder.py
  tests/
    test_tranformer_encoder_end_to_end.py
```

## Positional Encoding Behavior

The encoder supports `positional_encoding_type="rope"`, `"learned"`,
`"sinusoidal"`, or no positional encoding.

- Learned positional embeddings are added once to token embeddings at the model
  input: `(B, S, E) -> (B, S, E)`.
- Sinusoidal positional encodings are also added once at the model input:
  `(B, S, E) -> (B, S, E)`.
- Rotary positional embeddings are applied inside each self-attention layer to
  the projected query and key tensors: `(B, H, S, D) -> (B, H, S, D)`.
- Decoder self-attention can use RoPE in the same way. Decoder cross-attention
  remains plain multi-head attention because target and source sequence lengths
  may differ.

## Running Checks

Run the learned positional embedding demo:

```bash
python3 positional_encoding/learned_positional_embedding.py
```

Run the custom multi-head attention comparison against PyTorch:

```bash
python3 attention/tests/test_against_pytorch_mha.py
```

Run the transformer encoder pipeline test:

```bash
python3 transformer_blocks/tests/test_transformer_encoder_pipeline.py
```

Run the transformer decoder pipeline test:

```bash
python3 transformer_blocks/tests/test_transformer_decoder_pipeline.py
```

Run the transformer encoder end-to-end test:

```bash
PYTHONPATH=. python3 transformer_models/tests/test_tranformer_encoder_end_to_end.py
```

Compile all current Python files:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/pycache python3 -m py_compile \
  attention/masking.py \
  attention/multi_head_attention.py \
  attention/scaled_dot_product_attention.py \
  attention/tests/test_against_pytorch_mha.py \
  positional_encoding/learned_positional_embedding.py \
  positional_encoding/rotary_positional_embedding.py \
  positional_encoding/sinusoidal_positional_encoding.py \
  transformer_blocks/feed_forward_network.py \
  transformer_blocks/transformer_decoder_block.py \
  transformer_blocks/transformer_encoder_block.py \
  transformer_blocks/tests/test_transformer_decoder_pipeline.py \
  transformer_blocks/tests/test_transformer_encoder_pipeline.py \
  transformer_blocks/tests/test_transformer_shapes.py \
  transformer_models/transformer_encoder.py \
  transformer_models/tests/test_tranformer_encoder_end_to_end.py
```

If `pytest` is installed, the attention comparison can also be run with:

```bash
python3 -m pytest
```

## Notes

The GPT-2 style decoder-only model stack, tokenizer integration, training loop,
and text generation entry point are not implemented yet.
