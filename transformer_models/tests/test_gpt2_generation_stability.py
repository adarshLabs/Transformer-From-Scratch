import torch

from transformer_models.gpt2 import GPT2, GPT2Config


def test_generate_handles_invalid_logits_without_crashing():
    config = GPT2Config(
        n_layers=1,
        n_heads=1,
        d_model=8,
        vocab_size=16,
        block_size=8,
        dropout=0.0,
    )
    model = GPT2(config)

    with torch.no_grad():
        model.lm_head.weight.fill_(float("inf"))

    seed = torch.tensor([[1, 2, 3]], dtype=torch.long)
    out = model.generate(seed, max_new_tokens=2, temperature=0.8, top_k=4)

    assert out.shape == (1, 5)
    assert torch.isfinite(out).all()
