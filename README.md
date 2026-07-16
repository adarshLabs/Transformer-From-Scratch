# Transformer From Scratch

A clean PyTorch implementation of transformer building blocks, encoder and decoder components, and a GPT-style decoder-only model. The repository is intended as a learning-oriented reference for core transformer concepts such as attention, masking, positional encodings, block composition, and simple text generation training.

## What’s included

- Scaled dot-product attention
- Multi-head self-attention and cross-attention
- Causal, padding, and combined attention masks
- Positional encoding options: sinusoidal, learned, and rotary
- Feed-forward networks and transformer encoder/decoder blocks
- A small GPT-style decoder implementation with generation support
- A character-level tokenizer and a lightweight training loop for text generation

## Project structure

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

tokenizer/
  charactor_tokenizer.py

transformer_blocks/
  feed_forward_network.py
  transformer_decoder_block.py
  transformer_encoder_block.py
  tests/
    test_transformer_decoder_pipeline.py
    test_transformer_encoder_pipeline.py

transformer_models/
  gpt2.py
  gpt_decoder_block.py
  train.py
  transformer_encoder.py
  tests/
    test_tranformer_encoder_end_to_end.py
```

## Setup

Install the dependencies:

```bash
python3 -m pip install -r requirements.txt
```

## Quick start

Run the learned positional embedding demo:

```bash
python3 positional_encoding/learned_positional_embedding.py
```

Run the custom multi-head attention comparison against PyTorch:

```bash
python3 attention/tests/test_against_pytorch_mha.py
```

Run the transformer encoder end-to-end test:

```bash
PYTHONPATH=. python3 transformer_models/tests/test_tranformer_encoder_end_to_end.py
```

Run the GPT-style decoder demo:

```bash
python3 transformer_models/gpt2.py
```

Train the small decoder model on the bundled Shakespeare sample:

```bash
python3 transformer_models/train.py
```

Training outputs and checkpoints are written to the ignored `checkpoint/` directory.

## Testing

If pytest is available, you can run the test suite with:

```bash
python3 -m pytest
```

## Contributing

Contributions are welcome. If you want to improve the implementation, please open an issue or submit a pull request with a clear description of the change and any relevant tests.

## Notes

This repository is best suited for educational use and experimentation with transformer internals.
