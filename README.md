# GPT-2 Implementation

A from-scratch PyTorch implementation of core GPT-2 building blocks. The
repository currently focuses on attention and positional encoding components,
with small runnable demos and a comparison test against PyTorch's built-in
multi-head attention.

## Current Components

- Scaled dot-product attention
- Multi-head self-attention
- Causal, padding, and combined attention masks
- Sinusoidal positional encoding
- Learned positional embedding
- Test that compares the custom multi-head attention output with
  `torch.nn.MultiheadAttention`

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
  sinusoidal_positional_encoding.py
```

## Running Checks

Run the learned positional embedding demo:

```bash
python3 positional_encoding/learned_positional_embedding.py
```

Run the custom multi-head attention comparison against PyTorch:

```bash
python3 attention/tests/test_against_pytorch_mha.py
```

Compile all current Python files:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/pycache python3 -m py_compile \
  attention/masking.py \
  attention/multi_head_attention.py \
  attention/scaled_dot_product_attention.py \
  attention/tests/test_against_pytorch_mha.py \
  positional_encoding/learned_positional_embedding.py \
  positional_encoding/sinusoidal_positional_encoding.py
```

If `pytest` is installed, the attention comparison can also be run with:

```bash
python3 -m pytest
```

## Notes

The full GPT-2 model stack, transformer block, tokenizer integration, training
loop, and text generation entry point are not implemented yet.
