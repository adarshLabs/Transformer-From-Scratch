import torch

from transformer_models.gpt2 import GPT2, GPT2Config


def test_generate_supports_more_tokens_than_block_size():
    config = GPT2Config(vocab_size=32, block_size=8, n_layers=1, n_heads=2, d_model=16)
    model = GPT2(config)

    input_ids = torch.randint(0, config.vocab_size, (1, 4))

    for use_cache in (False, True):
        generated = model.generate(
            input_ids,
            max_new_tokens=16,
            temperature=1.0,
            top_k=8,
            use_cache=use_cache,
        )
        assert generated.shape[1] == input_ids.shape[1] + 16
