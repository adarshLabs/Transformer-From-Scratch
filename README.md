# Transformer From Scratch

A clean, educational PyTorch implementation of transformer internals. This repository builds the core pieces by hand: scaled dot-product attention, multi-head attention, masking, positional encodings, encoder/decoder blocks, and a compact GPT-style decoder-only language model that can be trained on small text datasets.

The goal is to make transformer mechanics readable and hackable without hiding the important parts behind a large framework.

## What Is Included

- Scaled dot-product attention
- Multi-head self-attention and cross-attention
- Causal, padding, and combined masks
- Sinusoidal, learned, and rotary positional encodings
- Feed-forward networks
- Transformer encoder and decoder blocks
- A GPT-style decoder-only model
- Tokenized dataset preprocessing with GPT-2 BPE via `tiktoken`
- Training, checkpointing, TensorBoard logging, and text generation scripts
- Focused tests for attention masks, PyTorch MHA parity, and transformer block shapes

## Project Structure

```text
attention/
  masking.py                         # causal, padding, and combined masks
  multi_head_attention.py            # custom multi-head attention
  scaled_dot_product_attention.py    # attention score/probability/value core
  tests/

positional_encoding/
  learned_positional_embedding.py
  rotary_positional_embedding.py
  sinusoidal_positional_encoding.py

scripts/
  preprocess.py                      # create train.bin, val.bin, and meta.pkl
  train.py                           # train the GPT-style decoder model
  generate.py                        # sample text from a saved checkpoint

tokenizer/
  character_tokenizer.py             # simple character tokenizer reference

transformer_blocks/
  feed_forward_network.py
  gpt_decoder_block.py
  transformer_decoder_block.py
  transformer_encoder_block.py
  tests/

transformer_models/
  gpt2.py                            # GPT-style decoder-only model
  transformer_encoder.py             # encoder model wrapper

data/raw/
  tiny_shakespeare.txt               # bundled sample dataset
```

Generated artifacts such as `data/processed/`, `checkpoint/`, and `runs/` are ignored by Git.

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

## Quick Start

Preprocess the bundled Tiny Shakespeare data:

```bash
python3 scripts/preprocess.py --dataset tiny_shakespeare
```

Train the GPT-style decoder:

```bash
python3 scripts/train.py --dataset tiny_shakespeare --max_steps 2000
```

Resume training from a checkpoint:

```bash
python3 scripts/train.py --dataset tiny_shakespeare --resume checkpoint/tiny_shakespeare/latest.pt
```

Generate text from the best checkpoint:

```bash
python3 scripts/generate.py --dataset tiny_shakespeare --checkpoint best.pt --prompt "The"
```

The generation script compares decoding with and without KV cache and prints the speedup.

## Optional Dataset

The preprocessing script also supports TinyStories from Hugging Face:

```bash
python3 scripts/preprocess.py --dataset tiny_stories
python3 scripts/train.py --dataset tiny_stories
```

This path uses the `datasets` package and may download data on first run.

## Useful Demos And Tests

Run the learned positional embedding demo:

```bash
python3 positional_encoding/learned_positional_embedding.py
```

Compare custom multi-head attention against PyTorch:

```bash
PYTHONPATH=. python3 attention/tests/test_against_pytorch_mha.py
```

Run the GPT model smoke test:

```bash
PYTHONPATH=. python3 transformer_models/gpt2.py
```

Run the full test suite:

```bash
PYTHONPATH=. python3 -m pytest
```

## Training Outputs

Training writes:

- checkpoints to `checkpoint/<dataset>/`
- TensorBoard logs to `runs/<dataset>/`
- processed token arrays and metadata to `data/processed/<dataset>/`

View TensorBoard logs with:

```bash
tensorboard --logdir runs
```

## Implementation Notes

The model intentionally keeps the architecture approachable. Attention masks are built explicitly, the GPT decoder block is composed from reusable attention and feed-forward modules, and the training loop exposes the usual language-modeling workflow: batches, next-token targets, validation loss, learning-rate scheduling, gradient clipping, checkpointing, and sampling.

KV caching is implemented for generation so the model can reuse previous keys and values during autoregressive decoding instead of recomputing the full context at every new token.

## Contributing

Contributions are welcome. Please include a clear description of the change and run the relevant tests before opening a pull request.

## License

This project is licensed under the MIT License.
