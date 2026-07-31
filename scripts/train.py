import argparse
import math
import os
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import tiktoken

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from transformer_models.gpt2 import GPT2, GPT2Config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", "--steps", dest="max_steps", type=int, default=2000)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_val_batches", type=int, default=50)
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--dataset", default="tiny_shakespeare")
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint to resume training from",
    )
    return parser.parse_args()


def load_data(path: Path):
    train_data = np.memmap(path / "train.bin", dtype=np.uint16, mode="r")
    val_data = np.memmap(path / "val.bin", dtype=np.uint16, mode="r")
    with open(path / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    return train_data, val_data, meta


def get_batch(data, block_size, batch_size, device):
    indices = torch.randint(len(data) - 1 - block_size, (batch_size,)).tolist()

    x = torch.stack(
        [torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in indices]
    )
    y = torch.stack(
        [
            torch.from_numpy(data[i + 1 : i + block_size + 1].astype(np.int64))
            for i in indices
        ]
    )

    return x.to(device), y.to(device)


### if model can't overfit a single batch, then there should be .
def sanity_check(model, block_size, device):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Fix ONE batch — sample once, reuse every step
    vocab_size = model.config.vocab_size
    x = torch.randint(0, vocab_size, (4, block_size), device=device)
    y = torch.randint(0, vocab_size, (4, block_size), device=device)

    loss_history = []
    for step in range(200):
        logits, loss, _ = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        if step % 20 == 0:
            print(f"  step {step:3d}:  loss {loss.item():.4f}")

    final_loss = loss_history[-1]
    status = "PASSED" if final_loss < 0.5 else "FAILED — check your implementation"
    print(f"\nFinal loss: {final_loss:.4f}  {status}\n")
    return loss_history


def get_lr(step, warmup_steps, max_lr, min_lr, max_steps):
    if step < warmup_steps:
        return max_lr * step / warmup_steps

    if step >= max_steps:
        return min_lr

    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(progress * math.pi))


def validation(model, val_data, device, args):
    model.eval()
    val_loss_history = []
    with torch.no_grad():
        for _ in range(args.num_val_batches):
            x, y = get_batch(val_data, args.block_size, args.batch_size, device=device)
            _, loss, _ = model(x, y)

            val_loss_history.append(loss.item())

    val_loss = np.mean(val_loss_history)
    print(f"Validation Loss: {val_loss:.4f} | " f"Perplexity: {math.exp(val_loss):.2f}")
    model.train()
    return val_loss


def train(model, train_data, val_data, tokenizer, device, args, config):
    model.train()
    os.makedirs("checkpoint", exist_ok=True)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.max_lr, weight_decay=0.1)
    start_step = 0
    best_val_loss = float("inf")
    if args.resume is not None:
        print(f"Loading checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimiser.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resuming from step {start_step}")
    if device.type == "cuda":
        model = torch.compile(model)
    lr_history = []
    loss_history = []
    for step in range(start_step, args.max_steps):
        lr = get_lr(step, args.warmup_steps, args.max_lr, args.min_lr, args.max_steps)
        for param_group in optimiser.param_groups:
            param_group["lr"] = lr

        x, y = get_batch(train_data, args.block_size, args.batch_size, device)
        _, loss, _ = model(x, y)

        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        optimiser.step()
        lr_history.append(float(lr))
        loss_history.append(float(loss.item()))
        if step % args.log_every == 0:
            print(f"step {step:4d} | loss {loss.item():.4f} | lr {lr:.6f} | grad_norm {grad_norm:.3f}")

        if step % args.save_every == 0 and step > 0:
            val_loss = validation(model, val_data, device, args)
            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimiser.state_dict(),
                "step": step,
                "best_val_loss": best_val_loss,
                "config": vars(config),
                "args": vars(args),
            }

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt["best_val_loss"] = best_val_loss
                torch.save(ckpt, "checkpoint/best.pt")

            torch.save(ckpt, "checkpoint/latest.pt")

            model.eval()
            prompt = "Once upon a time"
            seed = torch.tensor([tokenizer.encode(prompt)], device=device)
            with torch.no_grad():
                out = model.generate(seed, max_new_tokens=200, temperature=0.8, top_k=40)
            print(f"--- Text Generated at {step} ---")
            print(tokenizer.decode(out[0].tolist()))
            model.train()

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimiser.state_dict(),
            "step": args.max_steps,
            "best_val_loss": best_val_loss,
            "config": config.__dict__,
            "args": vars(args),
        },
        "checkpoint/final.pt",
    )

    # plt.figure(figsize=(8, 4))
    # plt.plot(range(len(lr_history)), lr_history, label="learning rate")
    # plt.xlabel("step")
    # plt.ylabel("learning rate")
    # plt.title("Learning rate schedule")
    # plt.legend()
    # plt.show()

    # plt.figure(figsize=(8, 4))
    # plt.plot(range(len(loss_history)), loss_history, label="loss")
    # plt.xlabel("step")
    # plt.ylabel("loss")
    # plt.title("Training loss")
    # plt.legend()
    # plt.show()


def main():
    args = parse_args()
    torch.manual_seed(1337)
    np.random.seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1337)
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # for local files
    path = Path("data/processed")
    train_data, val_data, meta = load_data(path / args.dataset)

    config = GPT2Config()
    config.block_size = args.block_size
    config.vocab_size = meta["vocab_size"]

    tokenizer = tiktoken.get_encoding(meta["tokenizer"])
    model = GPT2(config).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters : {num_params:,}")
    print("=" * 60)
    print(f"Dataset      : {args.dataset}")
    print(f"Device       : {device}")
    print(f"Vocabulary   : {config.vocab_size}")
    print(f"Block size   : {config.block_size}")
    print(f"Batch size   : {args.batch_size}")
    print(f"Steps        : {args.max_steps}")
    print(f"Train tokens : {len(train_data):,}")
    print(f"Val tokens   : {len(val_data):,}")
    print("=" * 60)

    train(model, train_data, val_data, tokenizer, device, args, config)


if __name__ == "__main__":
    main()
