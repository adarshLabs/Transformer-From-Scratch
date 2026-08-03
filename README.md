# Transformer From Scratch

A clean, educational PyTorch implementation of transformer internals. This repository builds the core pieces by hand: scaled dot-product attention, multi-head attention, masking, positional encodings, encoder/decoder blocks, and a compact GPT-style decoder-only language model.

The goal is to make transformer mechanics easy to read, inspect, and modify without hiding the important details behind a large framework.

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
```

For a larger corpus such as TinyStories, the default toy-sized transformer settings are usually too small. A stronger recipe is:

```bash
python3 scripts/train.py \
  --dataset tiny_stories \
  --n_layers 12 \
  --n_heads 12 \
  --d_model 768 \
  --expansion_factor 4 \
  --batch_size 24 \
  --block_size 256 \
  --max_steps 10000 \
  --warmup_steps 300 \
  --max_lr 3e-4 \
  --min_lr 3e-5 \
  --max_norm 1.0 \
  --log_every 100 \
  --save_every 1000 \
  --dropout 0.0
```

This uses a noticeably larger GPT-style model than the Shakespeare smoke test and is a much better fit for a larger dataset like TinyStories. The script also exposes the core architecture knobs directly if you want to tune the model further.

If you have a GPU or Apple Silicon hardware, this is the kind of setup that benefits most from it. The path uses the `datasets` package and may download data on first run.

## GPT-2 Learning Path

If your goal is to understand this repository's GPT-2 implementation without getting lost in the broader encoder/decoder transformer code, follow this path:

1. `tokenizer/character_tokenizer.py` - see how text is converted into token IDs.
2. `transformer_models/gpt2.py` - start here for the full GPT-style model definition.
3. `transformer_blocks/gpt_decoder_block.py` - understand one decoder block used inside GPT-2.
4. `attention/masking.py` - review the causal and padding masks used during attention.
5. `attention/multi_head_attention.py` and `attention/scaled_dot_product_attention.py` - inspect the attention mechanism that powers the model.
6. `positional_encoding/rotary_positional_embedding.py` - read the RoPE implementation used for position-aware attention.
7. `positional_encoding/sinusoidal_positional_encoding.py` and `positional_encoding/learned_positional_embedding.py` - compare the other positional encoding options.
8. `scripts/train.py` - follow the training loop and the data flow into the model.
9. `scripts/generate.py` - see how sampling and KV caching work at inference time.

You can safely skip `transformer_encoder.py` and the encoder-focused transformer block files unless you specifically want to study the encoder side of the project.

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
