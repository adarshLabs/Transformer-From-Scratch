import sys


def test_parse_args_allows_tinystories_recipe(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--dataset",
            "tiny_stories",
            "--n_layers",
            "12",
            "--n_heads",
            "12",
            "--d_model",
            "768",
            "--batch_size",
            "32",
            "--block_size",
            "512",
            "--max_steps",
            "50000",
            "--warmup_steps",
            "1000",
            "--max_lr",
            "6e-4",
            "--min_lr",
            "6e-5",
        ],
    )

    from scripts import train as train_module

    args = train_module.parse_args()

    assert args.dataset == "tiny_stories"
    assert args.n_layers == 12
    assert args.n_heads == 12
    assert args.d_model == 768
    assert args.batch_size == 32
    assert args.block_size == 512
    assert args.max_steps == 50000
    assert args.warmup_steps == 1000
    assert args.max_lr == 6e-4
    assert args.min_lr == 6e-5
